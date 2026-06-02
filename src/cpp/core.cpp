// chess-bote C++ core (nanobind extension module `chess_bote._core`).
//
// A native port of the engine's hot path: bitboard board representation,
// magic-bitboard attack generation, single-pass legal move generation, and
// make/unmake. Exposes perft so the whole stack can be validated against the
// pure-Python perft oracle (identical node counts) and benchmarked.
//
// Conventions match the Python implementation: LERF square indexing (a1 = 0),
// 6 piece-type bitboards + 2 color bitboards + a mailbox, and the 16-bit packed
// move encoding (flag << 12 | from << 6 | to).

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using Bitboard = uint64_t;
using Move = uint16_t;

// Colors.
constexpr int WHITE = 0, BLACK = 1;
// Piece types.
constexpr int PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5, NONE = 6;

// Squares (only the ones we name explicitly).
enum {
    A1 = 0, B1, C1, D1, E1, F1, G1, H1,
    A8 = 56, B8, C8, D8, E8, F8, G8, H8
};

// Castling-rights bit flags.
constexpr int WHITE_OO = 1, WHITE_OOO = 2, BLACK_OO = 4, BLACK_OOO = 8;

// Move flags.
constexpr int FQUIET = 0, FDOUBLE = 1, FKCASTLE = 2, FQCASTLE = 3, FCAPTURE = 4,
              FEP = 5, FPROMO_N = 8;

inline Move mkMove(int from, int to, int flag) {
    return (Move)((flag << 12) | (from << 6) | to);
}
inline int mFrom(Move m) { return (m >> 6) & 63; }
inline int mTo(Move m) { return m & 63; }
inline int mFlag(Move m) { return (m >> 12) & 15; }
inline bool mIsCapture(Move m) { return mFlag(m) & 4; }
inline bool mIsPromo(Move m) { return mFlag(m) & 8; }
inline int mPromoType(Move m) { return (mFlag(m) & 3) + KNIGHT; }

inline int popcount(Bitboard b) { return __builtin_popcountll(b); }
inline int lsb(Bitboard b) { return __builtin_ctzll(b); }
inline int popLsb(Bitboard& b) {
    int s = __builtin_ctzll(b);
    b &= b - 1;
    return s;
}

// --------------------------------------------------------------------------- //
// Precomputed tables
// --------------------------------------------------------------------------- //

Bitboard KNIGHT_ATT[64];
Bitboard KING_ATT[64];
Bitboard PAWN_ATT[2][64];
Bitboard BETWEEN[64][64];
Bitboard LINE[64][64];

Bitboard ROOK_MASK[64], BISHOP_MASK[64];
Bitboard ROOK_MAGIC[64], BISHOP_MAGIC[64];
int ROOK_SHIFT[64], BISHOP_SHIFT[64];
std::vector<Bitboard> ROOK_TABLE[64], BISHOP_TABLE[64];

int CASTLE_MASK[64];

// Zobrist keys (for the transposition table). Generated once at init; they
// only need to be self-consistent within the C++ core.
Bitboard ZOBRIST_PIECE[2][6][64];
Bitboard ZOBRIST_CASTLE[16];
Bitboard ZOBRIST_EP[8];
Bitboard ZOBRIST_SIDE;

inline bool inBounds(int f, int r) { return f >= 0 && f < 8 && r >= 0 && r < 8; }

const int ROOK_DIRS[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
const int BISHOP_DIRS[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

Bitboard slowSlide(int sq, Bitboard occ, const int dirs[4][2]) {
    Bitboard a = 0;
    int f = sq & 7, r = sq >> 3;
    for (int d = 0; d < 4; ++d) {
        int nf = f + dirs[d][0], nr = r + dirs[d][1];
        while (inBounds(nf, nr)) {
            int s = nr * 8 + nf;
            a |= 1ULL << s;
            if (occ & (1ULL << s)) break;
            nf += dirs[d][0];
            nr += dirs[d][1];
        }
    }
    return a;
}

Bitboard rookMaskOf(int sq) {
    int f = sq & 7, r = sq >> 3;
    Bitboard m = 0;
    for (int nr = r + 1; nr < 7; ++nr) m |= 1ULL << (nr * 8 + f);
    for (int nr = r - 1; nr > 0; --nr) m |= 1ULL << (nr * 8 + f);
    for (int nf = f + 1; nf < 7; ++nf) m |= 1ULL << (r * 8 + nf);
    for (int nf = f - 1; nf > 0; --nf) m |= 1ULL << (r * 8 + nf);
    return m;
}

Bitboard bishopMaskOf(int sq) {
    int f = sq & 7, r = sq >> 3;
    Bitboard m = 0;
    for (int d = 0; d < 4; ++d) {
        int nf = f + BISHOP_DIRS[d][0], nr = r + BISHOP_DIRS[d][1];
        while (nf > 0 && nf < 7 && nr > 0 && nr < 7) {
            m |= 1ULL << (nr * 8 + nf);
            nf += BISHOP_DIRS[d][0];
            nr += BISHOP_DIRS[d][1];
        }
    }
    return m;
}

// Enumerate every subset of `mask` (carry-rippler) into `out`.
void subsets(Bitboard mask, std::vector<Bitboard>& out) {
    out.clear();
    Bitboard sub = 0;
    do {
        out.push_back(sub);
        sub = (sub - mask) & mask;
    } while (sub != 0);
}

void findMagics(Bitboard (*maskFn)(int), const int dirs[4][2], Bitboard* magics,
                int* shifts, Bitboard* masks, std::vector<Bitboard>* tables) {
    std::mt19937_64 rng(0xC0FFEEULL);
    std::vector<Bitboard> subs, atts;
    for (int sq = 0; sq < 64; ++sq) {
        Bitboard mask = maskFn(sq);
        int bits = popcount(mask);
        int shift = 64 - bits;
        subsets(mask, subs);
        atts.resize(subs.size());
        for (size_t i = 0; i < subs.size(); ++i)
            atts[i] = slowSlide(sq, subs[i], dirs);

        std::vector<Bitboard> table(1ULL << bits);
        while (true) {
            Bitboard magic = rng() & rng() & rng();  // sparse candidate
            if (popcount((mask * magic) & 0xFF00000000000000ULL) < 6) continue;
            std::fill(table.begin(), table.end(), 0);
            std::vector<char> used(1ULL << bits, 0);
            bool ok = true;
            for (size_t i = 0; i < subs.size(); ++i) {
                size_t idx = (size_t)((subs[i] * magic) >> shift);
                if (!used[idx]) {
                    used[idx] = 1;
                    table[idx] = atts[i];
                } else if (table[idx] != atts[i]) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                magics[sq] = magic;
                shifts[sq] = shift;
                masks[sq] = mask;
                tables[sq] = std::move(table);
                break;
            }
        }
    }
}

inline Bitboard rookAtt(int sq, Bitboard occ) {
    occ &= ROOK_MASK[sq];
    return ROOK_TABLE[sq][(occ * ROOK_MAGIC[sq]) >> ROOK_SHIFT[sq]];
}
inline Bitboard bishopAtt(int sq, Bitboard occ) {
    occ &= BISHOP_MASK[sq];
    return BISHOP_TABLE[sq][(occ * BISHOP_MAGIC[sq]) >> BISHOP_SHIFT[sq]];
}
inline Bitboard queenAtt(int sq, Bitboard occ) {
    return rookAtt(sq, occ) | bishopAtt(sq, occ);
}

void initTables() {
    const int knightD[8][2] = {{1, 2},  {2, 1},  {2, -1},  {1, -2},
                               {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    const int kingD[8][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1},
                             {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
    for (int sq = 0; sq < 64; ++sq) {
        int f = sq & 7, r = sq >> 3;
        Bitboard kn = 0, kg = 0, pw = 0, pb = 0;
        for (int d = 0; d < 8; ++d) {
            int nf = f + knightD[d][0], nr = r + knightD[d][1];
            if (inBounds(nf, nr)) kn |= 1ULL << (nr * 8 + nf);
            nf = f + kingD[d][0]; nr = r + kingD[d][1];
            if (inBounds(nf, nr)) kg |= 1ULL << (nr * 8 + nf);
        }
        if (inBounds(f - 1, r + 1)) pw |= 1ULL << ((r + 1) * 8 + (f - 1));
        if (inBounds(f + 1, r + 1)) pw |= 1ULL << ((r + 1) * 8 + (f + 1));
        if (inBounds(f - 1, r - 1)) pb |= 1ULL << ((r - 1) * 8 + (f - 1));
        if (inBounds(f + 1, r - 1)) pb |= 1ULL << ((r - 1) * 8 + (f + 1));
        KNIGHT_ATT[sq] = kn;
        KING_ATT[sq] = kg;
        PAWN_ATT[WHITE][sq] = pw;
        PAWN_ATT[BLACK][sq] = pb;
    }

    // BETWEEN and LINE.
    for (int a = 0; a < 64; ++a) {
        int fa = a & 7, ra = a >> 3;
        for (int b = 0; b < 64; ++b) {
            BETWEEN[a][b] = 0;
            LINE[a][b] = 0;
            if (a == b) continue;
            int fb = b & 7, rb = b >> 3;
            bool sr = ra == rb, sf = fa == fb,
                 sd = std::abs(fa - fb) == std::abs(ra - rb);
            if (!(sr || sf || sd)) continue;
            int df = (fb > fa) - (fb < fa);
            int dr = (rb > ra) - (rb < ra);
            Bitboard bw = 0;
            int nf = fa + df, nr = ra + dr;
            while (nf != fb || nr != rb) {
                bw |= 1ULL << (nr * 8 + nf);
                nf += df;
                nr += dr;
            }
            BETWEEN[a][b] = bw;
            Bitboard ln = 1ULL << a;
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                nf = fa + df * sgn;
                nr = ra + dr * sgn;
                while (inBounds(nf, nr)) {
                    ln |= 1ULL << (nr * 8 + nf);
                    nf += df * sgn;
                    nr += dr * sgn;
                }
            }
            LINE[a][b] = ln;
        }
    }

    findMagics(rookMaskOf, ROOK_DIRS, ROOK_MAGIC, ROOK_SHIFT, ROOK_MASK, ROOK_TABLE);
    findMagics(bishopMaskOf, BISHOP_DIRS, BISHOP_MAGIC, BISHOP_SHIFT, BISHOP_MASK,
               BISHOP_TABLE);

    for (int i = 0; i < 64; ++i) CASTLE_MASK[i] = 15;
    CASTLE_MASK[E1] = 15 & ~(WHITE_OO | WHITE_OOO);
    CASTLE_MASK[H1] = 15 & ~WHITE_OO;
    CASTLE_MASK[A1] = 15 & ~WHITE_OOO;
    CASTLE_MASK[E8] = 15 & ~(BLACK_OO | BLACK_OOO);
    CASTLE_MASK[H8] = 15 & ~BLACK_OO;
    CASTLE_MASK[A8] = 15 & ~BLACK_OOO;

    std::mt19937_64 zrng(0x9E3779B97F4A7C15ULL);
    for (int c = 0; c < 2; ++c)
        for (int p = 0; p < 6; ++p)
            for (int s = 0; s < 64; ++s) ZOBRIST_PIECE[c][p][s] = zrng();
    for (int i = 0; i < 16; ++i) ZOBRIST_CASTLE[i] = zrng();
    for (int i = 0; i < 8; ++i) ZOBRIST_EP[i] = zrng();
    ZOBRIST_SIDE = zrng();
}

// --------------------------------------------------------------------------- //
// Board
// --------------------------------------------------------------------------- //

struct Board {
    Bitboard pieceBB[6] = {0, 0, 0, 0, 0, 0};
    Bitboard colorBB[2] = {0, 0};
    Bitboard occupied = 0;
    uint8_t mailbox[64];
    int side = WHITE;
    int castling = 0;
    int epSquare = -1;
    int halfmove = 0;
    int fullmove = 1;
    Bitboard zobrist = 0;

    Board() {
        for (int i = 0; i < 64; ++i) mailbox[i] = NONE;
    }

    inline Bitboard pieces(int color, int pt) const {
        return pieceBB[pt] & colorBB[color];
    }
    inline int kingSquare(int color) const {
        return lsb(pieceBB[KING] & colorBB[color]);
    }
    inline void put(int color, int pt, int sq) {
        Bitboard bb = 1ULL << sq;
        pieceBB[pt] |= bb;
        colorBB[color] |= bb;
        occupied |= bb;
        mailbox[sq] = (uint8_t)pt;
        zobrist ^= ZOBRIST_PIECE[color][pt][sq];
    }
    inline void remove(int color, int pt, int sq) {
        Bitboard bb = ~(1ULL << sq);
        pieceBB[pt] &= bb;
        colorBB[color] &= bb;
        occupied &= bb;
        mailbox[sq] = NONE;
        zobrist ^= ZOBRIST_PIECE[color][pt][sq];
    }
    inline void movePiece(int color, int pt, int from, int to) {
        remove(color, pt, from);
        put(color, pt, to);
    }
};

struct Undo {
    int captured;
    int castling;
    int epSquare;
    int halfmove;
    Bitboard zobrist;
};

void makeMove(Board& b, Move m, Undo& u) {
    int from = mFrom(m), to = mTo(m), flag = mFlag(m);
    int us = b.side, them = us ^ 1;
    int movingPt = b.mailbox[from];

    int captured = NONE;
    if (flag == FEP)
        captured = PAWN;
    else if (mIsCapture(m))
        captured = b.mailbox[to];

    u.captured = captured;
    u.castling = b.castling;
    u.epSquare = b.epSquare;
    u.halfmove = b.halfmove;
    u.zobrist = b.zobrist;

    if (b.epSquare != -1) {
        b.zobrist ^= ZOBRIST_EP[b.epSquare & 7];
        b.epSquare = -1;
    }

    if (flag == FEP) {
        int capSq = (us == WHITE) ? to - 8 : to + 8;
        b.remove(them, PAWN, capSq);
    } else if (captured != NONE) {
        b.remove(them, captured, to);
    }

    if (mIsPromo(m)) {
        b.remove(us, PAWN, from);
        b.put(us, mPromoType(m), to);
    } else {
        b.movePiece(us, movingPt, from, to);
    }

    if (flag == FKCASTLE)
        b.movePiece(us, ROOK, (us == WHITE) ? H1 : H8, (us == WHITE) ? F1 : F8);
    else if (flag == FQCASTLE)
        b.movePiece(us, ROOK, (us == WHITE) ? A1 : A8, (us == WHITE) ? D1 : D8);

    if (flag == FDOUBLE) {
        b.epSquare = (from + to) / 2;
        b.zobrist ^= ZOBRIST_EP[b.epSquare & 7];
    }

    int newCastling = b.castling & CASTLE_MASK[from] & CASTLE_MASK[to];
    if (newCastling != b.castling) {
        b.zobrist ^= ZOBRIST_CASTLE[b.castling] ^ ZOBRIST_CASTLE[newCastling];
        b.castling = newCastling;
    }

    if (movingPt == PAWN || captured != NONE)
        b.halfmove = 0;
    else
        b.halfmove++;
    if (us == BLACK) b.fullmove++;
    b.side = them;
    b.zobrist ^= ZOBRIST_SIDE;
}

void unmakeMove(Board& b, Move m, const Undo& u) {
    int us = b.side ^ 1, them = b.side;
    b.side = us;
    int from = mFrom(m), to = mTo(m), flag = mFlag(m);

    if (mIsPromo(m)) {
        b.remove(us, mPromoType(m), to);
        b.put(us, PAWN, from);
    } else {
        b.movePiece(us, b.mailbox[to], to, from);
    }

    if (flag == FKCASTLE)
        b.movePiece(us, ROOK, (us == WHITE) ? F1 : F8, (us == WHITE) ? H1 : H8);
    else if (flag == FQCASTLE)
        b.movePiece(us, ROOK, (us == WHITE) ? D1 : D8, (us == WHITE) ? A1 : A8);

    if (flag == FEP) {
        int capSq = (us == WHITE) ? to - 8 : to + 8;
        b.put(them, PAWN, capSq);
    } else if (u.captured != NONE) {
        b.put(them, u.captured, to);
    }

    b.castling = u.castling;
    b.epSquare = u.epSquare;
    b.halfmove = u.halfmove;
    b.zobrist = u.zobrist;
    if (us == BLACK) b.fullmove--;
}

void makeNullMove(Board& b, Undo& u) {
    int us = b.side;
    u.captured = NONE;
    u.castling = b.castling;
    u.epSquare = b.epSquare;
    u.halfmove = b.halfmove;
    u.zobrist = b.zobrist;
    if (b.epSquare != -1) {
        b.zobrist ^= ZOBRIST_EP[b.epSquare & 7];
        b.epSquare = -1;
    }
    b.side = us ^ 1;
    b.zobrist ^= ZOBRIST_SIDE;
    b.halfmove++;
    if (us == BLACK) b.fullmove++;
}

void unmakeNullMove(Board& b, const Undo& u) {
    int us = b.side ^ 1;
    b.side = us;
    b.castling = u.castling;
    b.epSquare = u.epSquare;
    b.halfmove = u.halfmove;
    b.zobrist = u.zobrist;
    if (us == BLACK) b.fullmove--;
}

// --------------------------------------------------------------------------- //
// Legal move generation (mirrors the Python algorithm)
// --------------------------------------------------------------------------- //

const int QUIET_PROMOS[4] = {11, 10, 9, 8};       // Q, R, B, N
const int CAP_PROMOS[4] = {15, 14, 13, 12};

inline void addPawn(int from, int to, int promoRank, bool capture, Move* out, int& n) {
    if ((to >> 3) == promoRank) {
        const int* flags = capture ? CAP_PROMOS : QUIET_PROMOS;
        for (int i = 0; i < 4; ++i) out[n++] = mkMove(from, to, flags[i]);
    } else {
        out[n++] = mkMove(from, to, capture ? FCAPTURE : FQUIET);
    }
}

int genLegal(const Board& b, Move* out) {
    int n = 0;
    int us = b.side, them = us ^ 1;
    int kingSq = b.kingSquare(us);
    Bitboard occ = b.occupied;
    Bitboard own = b.colorBB[us];
    Bitboard enemy = b.colorBB[them];

    Bitboard theirPawns = b.pieces(them, PAWN);
    Bitboard theirKnights = b.pieces(them, KNIGHT);
    Bitboard theirBQ = (b.pieceBB[BISHOP] | b.pieceBB[QUEEN]) & enemy;
    Bitboard theirRQ = (b.pieceBB[ROOK] | b.pieceBB[QUEEN]) & enemy;

    // Checkers.
    Bitboard checkers = (KNIGHT_ATT[kingSq] & theirKnights) |
                        (PAWN_ATT[us][kingSq] & theirPawns) |
                        (bishopAtt(kingSq, occ) & theirBQ) |
                        (rookAtt(kingSq, occ) & theirRQ);

    // King-danger squares (with our king removed so sliders x-ray it).
    Bitboard occNoKing = occ ^ (1ULL << kingSq);
    Bitboard danger = 0;
    {
        Bitboard p = theirPawns;
        while (p) danger |= PAWN_ATT[them][popLsb(p)];
        Bitboard k = theirKnights;
        while (k) danger |= KNIGHT_ATT[popLsb(k)];
        danger |= KING_ATT[b.kingSquare(them)];
        Bitboard bq = theirBQ;
        while (bq) danger |= bishopAtt(popLsb(bq), occNoKing);
        Bitboard rq = theirRQ;
        while (rq) danger |= rookAtt(popLsb(rq), occNoKing);
    }

    // King moves.
    Bitboard kt = KING_ATT[kingSq] & ~own & ~danger;
    while (kt) {
        int to = popLsb(kt);
        out[n++] = mkMove(kingSq, to, ((enemy >> to) & 1) ? FCAPTURE : FQUIET);
    }

    int numCheckers = popcount(checkers);
    if (numCheckers >= 2) return n;  // double check: king moves only

    Bitboard checkMask;
    if (numCheckers == 1) {
        int checkerSq = lsb(checkers);
        int cpt = b.mailbox[checkerSq];
        if (cpt == BISHOP || cpt == ROOK || cpt == QUEEN)
            checkMask = BETWEEN[kingSq][checkerSq] | checkers;
        else
            checkMask = checkers;
    } else {
        checkMask = ~0ULL;
        // Castling (only when not in check).
        int rights = b.castling;
        if (us == WHITE) {
            if ((rights & WHITE_OO) && !((occ >> F1) & 1) && !((occ >> G1) & 1) &&
                !((danger >> E1) & 1) && !((danger >> F1) & 1) && !((danger >> G1) & 1))
                out[n++] = mkMove(E1, G1, FKCASTLE);
            if ((rights & WHITE_OOO) && !((occ >> D1) & 1) && !((occ >> C1) & 1) &&
                !((occ >> B1) & 1) && !((danger >> E1) & 1) && !((danger >> D1) & 1) &&
                !((danger >> C1) & 1))
                out[n++] = mkMove(E1, C1, FQCASTLE);
        } else {
            if ((rights & BLACK_OO) && !((occ >> F8) & 1) && !((occ >> G8) & 1) &&
                !((danger >> E8) & 1) && !((danger >> F8) & 1) && !((danger >> G8) & 1))
                out[n++] = mkMove(E8, G8, FKCASTLE);
            if ((rights & BLACK_OOO) && !((occ >> D8) & 1) && !((occ >> C8) & 1) &&
                !((occ >> B8) & 1) && !((danger >> E8) & 1) && !((danger >> D8) & 1) &&
                !((danger >> C8) & 1))
                out[n++] = mkMove(E8, C8, FQCASTLE);
        }
    }

    // Pins.
    Bitboard pinned = 0;
    Bitboard snipers = (rookAtt(kingSq, 0) & theirRQ) | (bishopAtt(kingSq, 0) & theirBQ);
    while (snipers) {
        int sniper = popLsb(snipers);
        Bitboard blockers = BETWEEN[kingSq][sniper] & occ;
        if (blockers && (blockers & (blockers - 1)) == 0 && (blockers & own))
            pinned |= blockers;
    }

    // Knights.
    Bitboard kn = b.pieces(us, KNIGHT);
    while (kn) {
        int from = popLsb(kn);
        if ((pinned >> from) & 1) continue;
        Bitboard t = KNIGHT_ATT[from] & ~own & checkMask;
        while (t) {
            int to = popLsb(t);
            out[n++] = mkMove(from, to, ((enemy >> to) & 1) ? FCAPTURE : FQUIET);
        }
    }

    // Sliders.
    for (int pt = BISHOP; pt <= QUEEN; ++pt) {
        Bitboard pcs = b.pieces(us, pt);
        while (pcs) {
            int from = popLsb(pcs);
            Bitboard att = (pt == BISHOP)  ? bishopAtt(from, occ)
                           : (pt == ROOK)  ? rookAtt(from, occ)
                                           : queenAtt(from, occ);
            Bitboard t = att & ~own & checkMask;
            if ((pinned >> from) & 1) t &= LINE[kingSq][from];
            while (t) {
                int to = popLsb(t);
                out[n++] = mkMove(from, to, ((enemy >> to) & 1) ? FCAPTURE : FQUIET);
            }
        }
    }

    // Pawns.
    int forward = (us == WHITE) ? 8 : -8;
    int startRank = (us == WHITE) ? 1 : 6;
    int promoRank = (us == WHITE) ? 7 : 0;
    Bitboard pawns = b.pieces(us, PAWN);
    while (pawns) {
        int from = popLsb(pawns);
        Bitboard line = ((pinned >> from) & 1) ? LINE[kingSq][from] : ~0ULL;

        int to = from + forward;
        if (!((occ >> to) & 1) && ((line >> to) & 1)) {
            if ((checkMask >> to) & 1) addPawn(from, to, promoRank, false, out, n);
            if ((from >> 3) == startRank) {
                int to2 = to + forward;
                if (!((occ >> to2) & 1) && ((line >> to2) & 1) && ((checkMask >> to2) & 1))
                    out[n++] = mkMove(from, to2, FDOUBLE);
            }
        }

        Bitboard caps = PAWN_ATT[us][from];
        while (caps) {
            int cap = popLsb(caps);
            if ((enemy >> cap) & 1) {
                if (((line >> cap) & 1) && ((checkMask >> cap) & 1))
                    addPawn(from, cap, promoRank, true, out, n);
            } else if (cap == b.epSquare) {
                int captured = (us == WHITE) ? cap - 8 : cap + 8;
                Bitboard occ2 = (occ ^ (1ULL << from) ^ (1ULL << captured)) | (1ULL << cap);
                if (!(rookAtt(kingSq, occ2) & theirRQ) &&
                    !(bishopAtt(kingSq, occ2) & theirBQ))
                    out[n++] = mkMove(from, cap, FEP);
            }
        }
    }

    return n;
}

// --------------------------------------------------------------------------- //
// FEN + perft
// --------------------------------------------------------------------------- //

Board fromFen(const std::string& fen) {
    Board b;
    std::istringstream ss(fen);
    std::string placement, active, castling, ep;
    ss >> placement >> active >> castling >> ep;
    int half = 0, full = 1;
    ss >> half >> full;

    int rank = 7, file = 0;
    for (char c : placement) {
        if (c == '/') {
            rank--;
            file = 0;
        } else if (c >= '1' && c <= '8') {
            file += c - '0';
        } else {
            int color = (c >= 'A' && c <= 'Z') ? WHITE : BLACK;
            char lc = (char)std::tolower(c);
            int pt = lc == 'p'   ? PAWN
                     : lc == 'n' ? KNIGHT
                     : lc == 'b' ? BISHOP
                     : lc == 'r' ? ROOK
                     : lc == 'q' ? QUEEN
                                 : KING;
            b.put(color, pt, rank * 8 + file);
            file++;
        }
    }
    b.side = (active == "w") ? WHITE : BLACK;
    b.castling = 0;
    if (castling.find('K') != std::string::npos) b.castling |= WHITE_OO;
    if (castling.find('Q') != std::string::npos) b.castling |= WHITE_OOO;
    if (castling.find('k') != std::string::npos) b.castling |= BLACK_OO;
    if (castling.find('q') != std::string::npos) b.castling |= BLACK_OOO;
    if (ep != "-")
        b.epSquare = (ep[0] - 'a') + (ep[1] - '1') * 8;
    b.halfmove = half;
    b.fullmove = full;

    // Finish the hash: pieces were already XOR'd in by put(); add the rest.
    b.zobrist ^= ZOBRIST_CASTLE[b.castling];
    if (b.epSquare != -1) b.zobrist ^= ZOBRIST_EP[b.epSquare & 7];
    if (b.side == BLACK) b.zobrist ^= ZOBRIST_SIDE;
    return b;
}

uint64_t perftRec(Board& b, int depth) {
    Move moves[256];
    int n = genLegal(b, moves);
    if (depth == 1) return (uint64_t)n;
    uint64_t nodes = 0;
    Undo u;
    for (int i = 0; i < n; ++i) {
        makeMove(b, moves[i], u);
        nodes += perftRec(b, depth - 1);
        unmakeMove(b, moves[i], u);
    }
    return nodes;
}

uint64_t perft(const std::string& fen, int depth) {
    if (depth <= 0) return 1;
    Board b = fromFen(fen);
    return perftRec(b, depth);
}

std::string moveToUci(Move m) {
    int from = mFrom(m), to = mTo(m);
    std::string s;
    s += (char)('a' + (from & 7));
    s += (char)('1' + (from >> 3));
    s += (char)('a' + (to & 7));
    s += (char)('1' + (to >> 3));
    if (mIsPromo(m)) s += "pnbrqk"[mPromoType(m)];
    return s;
}

nb::dict perftDivide(const std::string& fen, int depth) {
    nb::dict result;
    Board b = fromFen(fen);
    Move moves[256];
    int n = genLegal(b, moves);
    Undo u;
    for (int i = 0; i < n; ++i) {
        makeMove(b, moves[i], u);
        uint64_t nodes = (depth > 1) ? perftRec(b, depth - 1) : 1;
        unmakeMove(b, moves[i], u);
        result[moveToUci(moves[i]).c_str()] = nodes;
    }
    return result;
}

// --------------------------------------------------------------------------- //
// Evaluation (material + piece-square tables; parameters set from Python)
// --------------------------------------------------------------------------- //

// Tapered evaluation parameters (set from Python; single source of truth).
constexpr int EVAL_MAX_PHASE = 24;
int EVAL_MG_VALUE[6] = {82, 337, 365, 477, 1025, 0};
int EVAL_EG_VALUE[6] = {94, 281, 297, 512, 936, 0};
int EVAL_PHASE[6] = {0, 1, 1, 2, 4, 0};
int EVAL_MG_PST[6][64] = {{0}};
int EVAL_EG_PST[6][64] = {{0}};

// Fixed values for MVV-LVA move ordering (independent of eval tuning).
const int MVVLVA_VALUE[6] = {100, 320, 330, 500, 900, 0};

void setEvalParams(const std::vector<int>& mgValues, const std::vector<int>& egValues,
                   const std::vector<std::vector<int>>& mgPst,
                   const std::vector<std::vector<int>>& egPst,
                   const std::vector<int>& phase) {
    for (int i = 0; i < 6; ++i) {
        EVAL_MG_VALUE[i] = mgValues[i];
        EVAL_EG_VALUE[i] = egValues[i];
        EVAL_PHASE[i] = phase[i];
        for (int sq = 0; sq < 64; ++sq) {
            EVAL_MG_PST[i][sq] = mgPst[i][sq];
            EVAL_EG_PST[i][sq] = egPst[i][sq];
        }
    }
}

int evaluateHandcrafted(const Board& b) {
    int mg = 0, eg = 0, phase = 0;
    Bitboard white = b.colorBB[WHITE], black = b.colorBB[BLACK];
    for (int pt = 0; pt < 6; ++pt) {
        int mgv = EVAL_MG_VALUE[pt], egv = EVAL_EG_VALUE[pt], ph = EVAL_PHASE[pt];
        const int* mgT = EVAL_MG_PST[pt];
        const int* egT = EVAL_EG_PST[pt];
        Bitboard w = b.pieceBB[pt] & white;
        while (w) {
            int s = popLsb(w);
            mg += mgv + mgT[s];
            eg += egv + egT[s];
            phase += ph;
        }
        Bitboard bl = b.pieceBB[pt] & black;
        while (bl) {
            int s = popLsb(bl) ^ 56;
            mg -= mgv + mgT[s];
            eg -= egv + egT[s];
            phase += ph;
        }
    }
    if (phase > EVAL_MAX_PHASE) phase = EVAL_MAX_PHASE;
    // Integer division truncates toward zero, matching the Python reference.
    int score = (mg * phase + eg * (EVAL_MAX_PHASE - phase)) / EVAL_MAX_PHASE;
    return (b.side == WHITE) ? score : -score;
}

// --------------------------------------------------------------------------- //
// NNUE evaluation
// --------------------------------------------------------------------------- //
//
// v1 recomputes the accumulator from the board on every eval ("refresh"). It is
// simple and correct, and serves as the oracle for the planned incremental
// (efficiently-updatable) version. Feature index = color*384 + piece*64 + sq.

constexpr int NN_HIDDEN = 256;
constexpr int NN_FEATURES = 768;
constexpr int NN_QA = 255;
constexpr int NN_QB = 64;

bool gUseNnue = false;
std::vector<int16_t> NN_FT_W;   // [NN_FEATURES * NN_HIDDEN], row-major (feature, hidden)
std::vector<int16_t> NN_FT_B;   // [NN_HIDDEN]
std::vector<int16_t> NN_OUT_W;  // [NN_HIDDEN]
int64_t NN_OUT_B = 0;           // already scaled by QA*QB

bool loadNnue(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int32_t hdr[4];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (hdr[0] != NN_HIDDEN || hdr[1] != NN_FEATURES || hdr[2] != NN_QA ||
        hdr[3] != NN_QB)
        return false;
    NN_FT_W.resize((size_t)NN_FEATURES * NN_HIDDEN);
    NN_FT_B.resize(NN_HIDDEN);
    NN_OUT_W.resize(NN_HIDDEN);
    f.read(reinterpret_cast<char*>(NN_FT_W.data()), NN_FT_W.size() * 2);
    f.read(reinterpret_cast<char*>(NN_FT_B.data()), NN_FT_B.size() * 2);
    f.read(reinterpret_cast<char*>(NN_OUT_W.data()), NN_OUT_W.size() * 2);
    f.read(reinterpret_cast<char*>(&NN_OUT_B), 8);
    if (!f) return false;
    gUseNnue = true;
    return true;
}

int evaluateNnue(const Board& b) {
    int32_t acc[NN_HIDDEN];
    for (int h = 0; h < NN_HIDDEN; ++h) acc[h] = NN_FT_B[h];
    for (int color = 0; color < 2; ++color) {
        Bitboard cbb = b.colorBB[color];
        int baseC = color * 384;
        for (int pt = 0; pt < 6; ++pt) {
            Bitboard bb = b.pieceBB[pt] & cbb;
            int base = baseC + pt * 64;
            while (bb) {
                int sq = popLsb(bb);
                const int16_t* w = &NN_FT_W[(size_t)(base + sq) * NN_HIDDEN];
                for (int h = 0; h < NN_HIDDEN; ++h) acc[h] += w[h];
            }
        }
    }
    int64_t out = 0;
    for (int h = 0; h < NN_HIDDEN; ++h) {
        int a = acc[h];
        if (a < 0) a = 0;
        else if (a > NN_QA) a = NN_QA;
        out += (int64_t)NN_OUT_W[h] * a;
    }
    int cp = (int)((out + NN_OUT_B) / ((int64_t)NN_QA * NN_QB));  // trunc toward zero
    return (b.side == WHITE) ? cp : -cp;
}

int evaluate(const Board& b) {
    return gUseNnue ? evaluateNnue(b) : evaluateHandcrafted(b);
}

int evaluateFen(const std::string& fen) { return evaluate(fromFen(fen)); }

// --------------------------------------------------------------------------- //
// Attack query (for check detection in search)
// --------------------------------------------------------------------------- //

bool isAttacked(const Board& b, int sq, int by) {
    if (PAWN_ATT[by ^ 1][sq] & b.pieces(by, PAWN)) return true;
    if (KNIGHT_ATT[sq] & b.pieces(by, KNIGHT)) return true;
    if (KING_ATT[sq] & b.pieces(by, KING)) return true;
    Bitboard occ = b.occupied;
    Bitboard bq = (b.pieceBB[BISHOP] | b.pieceBB[QUEEN]) & b.colorBB[by];
    if (bishopAtt(sq, occ) & bq) return true;
    Bitboard rq = (b.pieceBB[ROOK] | b.pieceBB[QUEEN]) & b.colorBB[by];
    if (rookAtt(sq, occ) & rq) return true;
    return false;
}

inline bool inCheck(const Board& b, int color) {
    return isAttacked(b, b.kingSquare(color), color ^ 1);
}

inline bool hasNonPawnMaterial(const Board& b, int color) {
    return (b.colorBB[color] & ~(b.pieceBB[PAWN] | b.pieceBB[KING])) != 0;
}

// --------------------------------------------------------------------------- //
// Search: negamax + alpha-beta, qsearch, ordering, PVS, null-move, TT
// --------------------------------------------------------------------------- //

constexpr int INF = 1000000000;
constexpr int MATE = 1000000;
constexpr int MATE_THRESHOLD = MATE - 1000;
constexpr int MAX_PLY = 256;
constexpr int NULL_R = 2;
constexpr int TT_EXACT = 0, TT_LOWER = 1, TT_UPPER = 2;

struct TTEntry {
    Bitboard key = 0;
    int score = 0;
    int depth = -1;
    int flag = 0;
    Move move = 0;
};

inline int scoreToTT(int score, int ply) {
    if (score >= MATE_THRESHOLD) return score + ply;
    if (score <= -MATE_THRESHOLD) return score - ply;
    return score;
}
inline int scoreFromTT(int score, int ply) {
    if (score >= MATE_THRESHOLD) return score - ply;
    if (score <= -MATE_THRESHOLD) return score + ply;
    return score;
}

inline int mvvLva(const Board& b, Move m) {
    int s = 0;
    if (mIsCapture(m)) {
        int victim = (mFlag(m) == FEP) ? PAWN : b.mailbox[mTo(m)];
        int attacker = b.mailbox[mFrom(m)];
        s = MVVLVA_VALUE[victim] * 16 - MVVLVA_VALUE[attacker];
    }
    if (mIsPromo(m)) s += MVVLVA_VALUE[mPromoType(m)];
    return s;
}

struct SearchResult {
    Move bestMove = 0;
    int score = 0;
    int depth = 0;
    uint64_t nodes = 0;
    double seconds = 0.0;
};

// Called once per completed iterative-deepening depth:
//   (depth, score, nodes, seconds, pv-as-uci-strings)
using InfoFn =
    std::function<void(int, int, uint64_t, double, const std::vector<std::string>&)>;

struct Searcher {
    uint64_t nodes = 0;
    bool timeUp = false;
    bool useDeadline = false;
    std::atomic<bool> stopRequested{false};
    std::chrono::steady_clock::time_point deadline;
    Move killers[MAX_PLY][2];
    int history[2][64][64];
    std::vector<TTEntry> tt;
    size_t ttMask;

    Searcher() : tt(1 << 20), ttMask((1 << 20) - 1) {}

    void clearTT() { std::fill(tt.begin(), tt.end(), TTEntry{}); }

    // Reconstruct the principal variation by following best moves in the TT.
    // Done only once per completed depth (~10-15 times per search), so it adds
    // no cost to the hot search loop. Verifies legality and guards cycles so a
    // hash collision can never produce a bogus or infinite line.
    std::vector<std::string> reconstructPV(Board b, int maxLen) {
        std::vector<std::string> pv;
        std::vector<Bitboard> seen;
        Undo u;
        for (int i = 0; i < maxLen && i < 64; ++i) {
            TTEntry& e = tt[b.zobrist & ttMask];
            if (e.key != b.zobrist || e.move == 0) break;
            Move m = e.move;
            Move legal[256];
            int n = genLegal(b, legal);
            bool ok = false;
            for (int j = 0; j < n; ++j)
                if (legal[j] == m) { ok = true; break; }
            if (!ok) break;
            bool cycle = false;
            for (Bitboard k : seen)
                if (k == b.zobrist) { cycle = true; break; }
            if (cycle) break;
            seen.push_back(b.zobrist);
            pv.push_back(moveToUci(m));
            makeMove(b, m, u);
        }
        return pv;
    }

    inline void checkpoint() {
        if ((++nodes & 2047) == 0) {
            if (stopRequested.load(std::memory_order_relaxed))
                timeUp = true;
            else if (useDeadline && std::chrono::steady_clock::now() >= deadline)
                timeUp = true;
        }
    }

    void onCutoff(Move m, int side, int depth, int ply) {
        if (mIsCapture(m) || mIsPromo(m)) return;
        if (killers[ply][0] != m) {
            killers[ply][1] = killers[ply][0];
            killers[ply][0] = m;
        }
        history[side][mFrom(m)][mTo(m)] += depth * depth;
    }

    void orderMoves(const Board& b, Move* moves, int n, int ply, Move first) {
        std::pair<int, Move> buf[256];
        const int* hist0 = history[b.side][0];  // base; index per-from below
        for (int i = 0; i < n; ++i) {
            Move m = moves[i];
            int sc;
            if (m == first) {
                sc = 1 << 30;
            } else if (mIsCapture(m) || mIsPromo(m)) {
                sc = 1000000 + mvvLva(b, m);
            } else if (m == killers[ply][0]) {
                sc = 900000;
            } else if (m == killers[ply][1]) {
                sc = 800000;
            } else {
                sc = history[b.side][mFrom(m)][mTo(m)];
            }
            buf[i] = {sc, m};
        }
        std::sort(buf, buf + n,
                  [](const std::pair<int, Move>& a, const std::pair<int, Move>& c) {
                      return a.first > c.first;
                  });
        for (int i = 0; i < n; ++i) moves[i] = buf[i].second;
        (void)hist0;
    }

    int quiescence(Board& b, int alpha, int beta) {
        checkpoint();
        int stand = evaluate(b);
        if (stand >= beta) return beta;
        if (stand > alpha) alpha = stand;

        Move moves[256];
        int n = genLegal(b, moves);
        // Order only the tactical moves by MVV-LVA.
        std::pair<int, Move> buf[256];
        int m = 0;
        for (int i = 0; i < n; ++i) {
            if (mIsCapture(moves[i]) || mIsPromo(moves[i]))
                buf[m++] = {mvvLva(b, moves[i]), moves[i]};
        }
        std::sort(buf, buf + m,
                  [](const std::pair<int, Move>& a, const std::pair<int, Move>& c) {
                      return a.first > c.first;
                  });
        Undo u;
        for (int i = 0; i < m; ++i) {
            makeMove(b, buf[i].second, u);
            int score = -quiescence(b, -beta, -alpha);
            unmakeMove(b, buf[i].second, u);
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }
        return alpha;
    }

    int negamax(Board& b, int depth, int alpha, int beta, int ply) {
        checkpoint();
        if (timeUp) return 0;

        int alphaOrig = alpha;
        Bitboard key = b.zobrist;
        Move ttMove = 0;
        TTEntry& e = tt[key & ttMask];
        if (e.key == key) {
            ttMove = e.move;
            if (ply > 0 && e.depth >= depth) {
                int s = scoreFromTT(e.score, ply);
                if (e.flag == TT_EXACT) return s;
                if (e.flag == TT_LOWER && s >= beta) return s;
                if (e.flag == TT_UPPER && s <= alpha) return s;
            }
        }

        if (depth == 0) return quiescence(b, alpha, beta);

        bool checked = inCheck(b, b.side);

        bool isPv = beta - alpha > 1;
        if (!isPv && !checked && depth >= 3 && beta < MATE_THRESHOLD &&
            hasNonPawnMaterial(b, b.side)) {
            Undo u;
            makeNullMove(b, u);
            int nullScore = -negamax(b, depth - 1 - NULL_R, -beta, -beta + 1, ply + 1);
            unmakeNullMove(b, u);
            if (timeUp) return 0;
            if (nullScore >= beta) return beta;
        }

        Move moves[256];
        int n = genLegal(b, moves);
        if (n == 0) return checked ? -MATE + ply : 0;

        orderMoves(b, moves, n, ply, ttMove);
        int side = b.side;

        int best = -INF;
        Move bestMove = moves[0];
        Undo u;
        for (int i = 0; i < n; ++i) {
            Move m = moves[i];
            makeMove(b, m, u);
            int score;
            if (i == 0) {
                score = -negamax(b, depth - 1, -beta, -alpha, ply + 1);
            } else {
                score = -negamax(b, depth - 1, -alpha - 1, -alpha, ply + 1);
                if (alpha < score && score < beta)
                    score = -negamax(b, depth - 1, -beta, -alpha, ply + 1);
            }
            unmakeMove(b, m, u);
            if (timeUp) return 0;
            if (score > best) {
                best = score;
                bestMove = m;
                if (best > alpha) alpha = best;
                if (alpha >= beta) {
                    onCutoff(m, side, depth, ply);
                    break;
                }
            }
        }

        int flag = (best <= alphaOrig) ? TT_UPPER : (best >= beta) ? TT_LOWER : TT_EXACT;
        if (e.key != key || depth >= e.depth) {
            e.key = key;
            e.depth = depth;
            e.score = scoreToTT(best, ply);
            e.flag = flag;
            e.move = bestMove;
        }
        return best;
    }

    // Returns the root score; writes the best move to outMove.
    int searchRoot(Board& b, int depth, Move orderHint, Move& outMove) {
        Move moves[256];
        int n = genLegal(b, moves);
        if (n == 0) {
            outMove = 0;
            return inCheck(b, b.side) ? -MATE : 0;
        }
        TTEntry& e = tt[b.zobrist & ttMask];
        Move ttMove = (e.key == b.zobrist && e.move) ? e.move : orderHint;
        orderMoves(b, moves, n, 0, ttMove);

        int alpha = -INF;
        Move localBest = moves[0];
        Undo u;
        for (int i = 0; i < n; ++i) {
            Move m = moves[i];
            makeMove(b, m, u);
            int score;
            if (i == 0) {
                score = -negamax(b, depth - 1, -INF, -alpha, 1);
            } else {
                score = -negamax(b, depth - 1, -alpha - 1, -alpha, 1);
                if (score > alpha) score = -negamax(b, depth - 1, -INF, -alpha, 1);
            }
            unmakeMove(b, m, u);
            if (timeUp) break;
            if (score > alpha) {
                alpha = score;
                localBest = m;
            }
        }
        // The root is exact; cache it for the next iteration's ordering.
        e.key = b.zobrist;
        e.depth = depth;
        e.score = scoreToTT(alpha, 0);
        e.flag = TT_EXACT;
        e.move = localBest;
        outMove = localBest;
        return alpha;
    }

    SearchResult go(Board& b, int maxDepth, int movetimeMs, const InfoFn& onInfo) {
        if (maxDepth > MAX_PLY - 16) maxDepth = MAX_PLY - 16;  // keep ply in bounds
        if (maxDepth < 1) maxDepth = 1;
        nodes = 0;
        timeUp = false;
        stopRequested.store(false, std::memory_order_relaxed);
        useDeadline = movetimeMs > 0;
        if (useDeadline)
            deadline = std::chrono::steady_clock::now() +
                       std::chrono::milliseconds(movetimeMs);
        std::fill(&killers[0][0], &killers[0][0] + MAX_PLY * 2, (Move)0);
        std::fill(&history[0][0][0], &history[0][0][0] + 2 * 64 * 64, 0);

        auto start = std::chrono::steady_clock::now();
        Move bestMove = 0;
        int bestScore = 0;
        int completed = 0;
        Move hint = 0;

        for (int depth = 1; depth <= maxDepth; ++depth) {
            Move move;
            int score = searchRoot(b, depth, hint, move);
            if (timeUp) break;
            bestMove = move;
            bestScore = score;
            completed = depth;
            hint = move;
            if (onInfo) {
                double secs = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - start)
                                  .count();
                onInfo(depth, bestScore, nodes, secs, reconstructPV(b, depth));
            }
            if (std::abs(bestScore) >= MATE_THRESHOLD) break;
            if (useDeadline && std::chrono::steady_clock::now() >= deadline) break;
        }

        double seconds = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - start)
                             .count();
        SearchResult r;
        r.bestMove = bestMove;
        r.score = bestScore;
        r.depth = completed;
        r.nodes = nodes;
        r.seconds = seconds;
        return r;
    }
};

Searcher gSearcher;

nb::dict searchFen(const std::string& fen, int maxDepth, int movetimeMs,
                   nb::object infoCallback) {
    Board b = fromFen(fen);

    // Optional per-depth callback. The search runs with the GIL released, so the
    // callback reacquires it briefly (only ~once per depth) to call into Python.
    InfoFn onInfo;
    if (!infoCallback.is_none()) {
        onInfo = [&infoCallback](int depth, int score, uint64_t nodes, double secs,
                                 const std::vector<std::string>& pv) {
            nb::gil_scoped_acquire acquire;
            nb::dict info;
            info["depth"] = depth;
            info["score"] = score;
            info["nodes"] = nodes;
            info["time"] = secs;
            nb::list pvlist;
            for (const auto& s : pv) pvlist.append(s);
            info["pv"] = pvlist;
            infoCallback(info);
        };
    }

    SearchResult r;
    {
        // Release the GIL so a UCI `stop` from another thread can interrupt us.
        nb::gil_scoped_release release;
        r = gSearcher.go(b, maxDepth, movetimeMs, onInfo);
    }
    nb::dict d;
    d["bestmove"] = r.bestMove ? moveToUci(r.bestMove) : std::string("0000");
    d["score"] = r.score;
    d["depth"] = r.depth;
    d["nodes"] = r.nodes;
    d["time"] = r.seconds;
    return d;
}

void stopSearch() { gSearcher.stopRequested.store(true, std::memory_order_relaxed); }

void newGame() { gSearcher.clearTT(); }

}  // namespace

NB_MODULE(_core, m) {
    initTables();
    m.doc() = "chess-bote native core";
    m.def("perft", &perft, "fen"_a, "depth"_a,
          "Count legal-move-tree leaf nodes to the given depth.");
    m.def("perft_divide", &perftDivide, "fen"_a, "depth"_a,
          "Per-root-move perft node counts (UCI move -> nodes).");
    m.def("set_eval_params", &setEvalParams, "mg_values"_a, "eg_values"_a,
          "mg_pst"_a, "eg_pst"_a, "phase"_a,
          "Load tapered eval weights: midgame/endgame piece values, 6x64 "
          "midgame/endgame piece-square tables, and per-piece phase weights.");
    m.def("evaluate", &evaluateFen, "fen"_a,
          "Static evaluation (centipawns, relative to side to move).");
    m.def("load_nnue", &loadNnue, "path"_a,
          "Load a quantized NNUE from disk and switch eval to it. Returns success.");
    m.def("use_nnue", [](bool on) { gUseNnue = on; }, "on"_a,
          "Toggle between NNUE (true) and hand-crafted (false) evaluation.");
    m.def("nnue_active", []() { return gUseNnue; }, "Is NNUE eval active?");
    m.def("search", &searchFen, "fen"_a, "max_depth"_a, "movetime_ms"_a = 0,
          "info_callback"_a = nb::none(),
          "Search a position; returns {bestmove, score, depth, nodes, time}. "
          "If info_callback is given, it is called once per depth with a dict "
          "{depth, score, nodes, time, pv}.");
    m.def("stop", &stopSearch, "Request the running search to stop (thread-safe).");
    m.def("new_game", &newGame, "Clear the transposition table.");
}
