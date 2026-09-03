-- 文件已创建 tNavigator v22.1-1754-g0c53512efc0
-- 版权 (C) Rock Flow Dynamics 2005-2022.
-- 版权所有。

--RPTSCHED
-- 'WELLS=2' 'SUMMARY=2' 'fip=3' 'RESTART=1' 'WELSPECS' 'CPU=2' /

/


WELSPECS
-- wname   gname iw jw ref.depth phase radius inflow_flag shut-in_flag crossflow_ability_flag PVT_n density_type filt.tbl reserved reserved well_model_type polymer_mixing_tbl.n
   'WELL1' 1*    26 26 1*        WATER /
   'WELL2' 1*    48 48 1*        WATER /
   'WELL3' 1*    48  4 1*        WATER /
   'WELL4' 1*     4  4 1*        WATER /
   'WELL5' 1*     4 48 1*        WATER /
   'WELL6' 1*    48 26 1*        WATER /
   'WELL7' 1*    26  4 1*        WATER /
   'WELL8' 1*    26 48 1*        WATER /
   'WELL9' 1*     4 26 1*        WATER /
/

COMPDAT
-- wname   iw jw kw1 kw2 status filt.tbl pi                diameter Kh                skin D-factor direction r0
   'WELL1' 26 26  1   1  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL1' 26 26  2   2  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL1' 26 26  3   3  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL1' 26 26  4   4  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL1' 26 26  5   5  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL1' 26 26  6   6  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL1' 26 26  7   7  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL1' 26 26  8   8  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL1' 26 26  9   9  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL1' 26 26 10  10  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL1' 26 26 11  11  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL1' 26 26 12  12  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL1' 26 26 13  13  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL1' 26 26 14  14  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL1' 26 26 15  15  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL1' 26 26 16  16  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL1' 26 26 17  17  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL1' 26 26 18  18  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL1' 26 26 19  19  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL1' 26 26 20  20  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
/
/
/

COMPORD
-- wname   sort
   'WELL1' INPUT /
/

COMPDAT
-- wname   iw jw kw1 kw2 status filt.tbl pi               diameter Kh  skin D-factor direction r0
   'WELL2' 48 48  1   1  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL2' 48 48  2   2  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL2' 48 48  3   3  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL2' 48 48  4   4  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL2' 48 48  5   5  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL2' 48 48  6   6  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL2' 48 48  7   7  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL2' 48 48  8   8  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL2' 48 48  9   9  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL2' 48 48 10  10  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL2' 48 48 11  11  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL2' 48 48 12  12  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL2' 48 48 13  13  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL2' 48 48 14  14  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL2' 48 48 15  15  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL2' 48 48 16  16  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL2' 48 48 17  17  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL2' 48 48 18  18  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL2' 48 48 19  19  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL2' 48 48 20  20  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
/
/
/

COMPORD
-- wname   sort
   'WELL2' INPUT /
/

COMPDAT
-- wname   iw jw kw1 kw2 status filt.tbl pi               diameter Kh  skin D-factor direction r0
   'WELL3' 48 4   1   1  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL3' 48 4   2   2  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL3' 48 4   3   3  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL3' 48 4   4   4  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL3' 48 4   5   5  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL3' 48 4   6   6  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL3' 48 4   7   7  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL3' 48 4   8   8  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL3' 48 4   9   9  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL3' 48 4  10  10  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL3' 48 4  11  11  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL3' 48 4  12  12  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL3' 48 4  13  13  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL3' 48 4  14  14  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL3' 48 4  15  15  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL3' 48 4  16  16  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL3' 48 4  17  17  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL3' 48 4  18  18  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL3' 48 4  19  19  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL3' 48 4  20  20  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
/
/
/

COMPORD
-- wname   sort
   'WELL3' INPUT /
/

COMPDAT
-- wname   iw jw kw1 kw2 status filt.tbl pi               diameter Kh  skin D-factor direction r0
   'WELL4' 4  4   1   1  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL4' 4  4   2   2  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL4' 4  4   3   3  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL4' 4  4   4   4  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL4' 4  4   5   5  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL4' 4  4   6   6  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL4' 4  4   7   7  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL4' 4  4   8   8  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL4' 4  4   9   9  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL4' 4  4  10  10  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL4' 4  4  11  11  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL4' 4  4  12  12  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL4' 4  4  13  13  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL4' 4  4  14  14  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL4' 4  4  15  15  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL4' 4  4  16  16  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL4' 4  4  17  17  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL4' 4  4  18  18  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL4' 4  4  19  19  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL4' 4  4  20  20  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
/
/
/

COMPORD
-- wname   sort
   'WELL4' INPUT /
/

COMPDAT
-- wname   iw jw kw1 kw2 status filt.tbl pi               diameter Kh  skin D-factor direction r0
   'WELL5' 4  48  1   1  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL5' 4  48  2   2  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL5' 4  48  3   3  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL5' 4  48  4   4  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL5' 4  48  5   5  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL5' 4  48  6   6  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL5' 4  48  7   7  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL5' 4  48  8   8  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL5' 4  48  9   9  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL5' 4  48 10  10  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL5' 4  48 11  11  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL5' 4  48 12  12  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL5' 4  48 13  13  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL5' 4  48 14  14  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL5' 4  48 15  15  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL5' 4  48 16  16  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL5' 4  48 17  17  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL5' 4  48 18  18  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL5' 4  48 19  19  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL5' 4  48 20  20  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
/

COMPORD
-- wname   sort
   'WELL5' INPUT /
/

COMPDAT
-- wname   iw jw kw1 kw2 status filt.tbl pi                diameter Kh                skin D-factor direction r0
   'WELL6' 48 26  1   1  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL6' 48 26  2   2  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL6' 48 26  3   3  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL6' 48 26  4   4  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL6' 48 26  5   5  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL6' 48 26  6   6  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL6' 48 26  7   7  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL6' 48 26  8   8  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL6' 48 26  9   9  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL6' 48 26 10  10  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL6' 48 26 11  11  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL6' 48 26 12  12  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL6' 48 26 13  13  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL6' 48 26 14  14  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL6' 48 26 15  15  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL6' 48 26 16  16  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL6' 48 26 17  17  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL6' 48 26 18  18  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL6' 48 26 19  19  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL6' 48 26 20  20  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
/
/
/

COMPORD
-- wname   sort
   'WELL6' INPUT /
/

COMPDAT
-- wname   iw jw kw1 kw2 status filt.tbl pi               diameter Kh  skin D-factor direction r0
   'WELL7' 26 4   1   1  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL7' 26 4   2   2  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL7' 26 4   3   3  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL7' 26 4   4   4  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL7' 26 4   5   5  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL7' 26 4   6   6  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL7' 26 4   7   7  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL7' 26 4   8   8  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL7' 26 4   9   9  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL7' 26 4  10  10  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL7' 26 4  11  11  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL7' 26 4  12  12  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL7' 26 4  13  13  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL7' 26 4  14  14  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL7' 26 4  15  15  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL7' 26 4  16  16  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL7' 26 4  17  17  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL7' 26 4  18  18  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL7' 26 4  19  19  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL7' 26 4  20  20  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
/
/
/

COMPORD
-- wname   sort
   'WELL7' INPUT /
/

COMPDAT
-- wname   iw jw kw1 kw2 status filt.tbl pi               diameter Kh  skin D-factor direction r0
   'WELL8' 26 48  1   1  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL8' 26 48  2   2  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL8' 26 48  3   3  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL8' 26 48  4   4  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL8' 26 48  5   5  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL8' 26 48  6   6  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL8' 26 48  7   7  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL8' 26 48  8   8  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL8' 26 48  9   9  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL8' 26 48 10  10  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL8' 26 48 11  11  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL8' 26 48 12  12  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL8' 26 48 13  13  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL8' 26 48 14  14  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL8' 26 48 15  15  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL8' 26 48 16  16  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL8' 26 48 17  17  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL8' 26 48 18  18  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL8' 26 48 19  19  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
   'WELL8' 26 48 20  20  OPEN   1*       3.45514069382663 0.16     250 3*                      3.8608030252785499 /
/
/
/
/

COMPORD
-- wname   sort
   'WELL8' INPUT /
/

COMPDAT
-- wname   iw jw kw1 kw2 status filt.tbl pi                diameter Kh                skin D-factor direction r0
   'WELL9' 4  26  1   1  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL9' 4  26  2   2  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL9' 4  26  3   3  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL9' 4  26  4   4  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL9' 4  26  5   5  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL9' 4  26  6   6  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL9' 4  26  7   7  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL9' 4  26  8   8  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL9' 4  26  9   9  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL9' 4  26 10  10  OPEN   1*       7.592562628596567 0.16     549.3671098663474 3*
                      3.8608030252785499 /
   'WELL9' 4  26 11  11  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL9' 4  26 12  12  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL9' 4  26 13  13  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL9' 4  26 14  14  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL9' 4  26 15  15  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL9' 4  26 16  16  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL9' 4  26 17  17  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL9' 4  26 18  18  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL9' 4  26 19  19  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
   'WELL9' 4  26 20  20  OPEN   1*       3.45514069382663  0.16     250               3*
                      3.8608030252785499 /
/
/

COMPORD
-- wname   sort
   'WELL9' INPUT /
   'WELL1' INPUT /
   'WELL2' INPUT /
   'WELL3' INPUT /
   'WELL4' INPUT /
   'WELL5' INPUT /
   'WELL6' INPUT /
   'WELL7' INPUT /
   'WELL8' INPUT /
   'WELL9' INPUT /
/

WCONHIST
-- wname   status control orate   wrate  grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    7384.14 836.1  769497.94 3*            100 /
   'WELL2' OPEN   LRAT    2507.9  684.83 251511.21 3*            100 /
   'WELL3' OPEN   LRAT    2469.07 678.46 246742.7  3*            100 /
   'WELL4' OPEN   LRAT    2405.54 668.19 239048.83 3*            100 /
   'WELL5' OPEN   LRAT    2446.56 675.25 244064.25 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   11012.53          350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN    6239.37          350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN    6383.94          350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   10740.98          350 6*
                                                                         RATE    /
/


COMPORD
-- wname   sort
   'WELL1' INPUT /
   'WELL2' INPUT /
   'WELL3' INPUT /
   'WELL4' INPUT /
   'WELL5' INPUT /
   'WELL6' INPUT /
   'WELL7' INPUT /
   'WELL8' INPUT /
   'WELL9' INPUT /
   'WELL1' INPUT /
   'WELL2' INPUT /
   'WELL3' INPUT /
   'WELL4' INPUT /
   'WELL5' INPUT /
   'WELL6' INPUT /
   'WELL7' INPUT /
   'WELL8' INPUT /
   'WELL9' INPUT /
/

DATES
08 AUG 2025 /
/

WCONHIST
-- wname   status control orate   wrate  grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    8822.34 886.9  916063.51 3*            100 /
   'WELL2' OPEN   LRAT    2907.51 692.09 305633.98 3*            100 /
   'WELL3' OPEN   LRAT    2843.77 678.68 298372.37 3*            100 /
   'WELL4' OPEN   LRAT    2744.65 661.49 286605.82 3*            100 /
   'WELL5' OPEN   LRAT    2814.68 673.42 295001.86 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   10553.72          350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN    5973.53          350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN    6136.16          350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   10262.35          350 6*
                                                                         RATE    /
/


DATES
15 AUG 2025 /
/

WCONHIST
-- wname   status control orate   wrate  grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    9020.82 907.01 935694.15 3*            100 /
   'WELL2' OPEN   LRAT    2997.58 695.31 313722.19 3*            100 /
   'WELL3' OPEN   LRAT    2925.85 675.89 306442.24 3*            100 /
   'WELL4' OPEN   LRAT    2819.81 653.86 295280.27 3*            100 /
   'WELL5' OPEN   LRAT    2898.84 670.11 303713.89 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   10383.4           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN    5925.08          350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN    6112.22          350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   10088.77          350 6*
                                                                         RATE    /
/



DATES
22 AUG 2025 /
/

WCONHIST
-- wname   status control orate   wrate  grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    8975.32 920.27 930796.44 3*            100 /
   'WELL2' OPEN   LRAT    3013.22 695.97 313911.33 3*            100 /
   'WELL3' OPEN   LRAT    2938.14 675.78 306308.95 3*            100 /
   'WELL4' OPEN   LRAT    2832.53 649.9  295423.25 3*            100 /
   'WELL5' OPEN   LRAT    2914.65 668.96 303986.27 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   10269.29          350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN    5907.38          350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN    6110.17          350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN    9976.01          350 6*
                                                                         RATE    /
/


DATES
29 AUG 2025 /
/

WCONHIST
-- wname   status control orate   wrate  grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    8932.18 925.85 926289.1  3*            100 /
   'WELL2' OPEN   LRAT    3012.13 696.66 313471.85 3*            100 /
   'WELL3' OPEN   LRAT    2936.16 676.26 305770.13 3*            100 /
   'WELL4' OPEN   LRAT    2830.35 649.7  294873.42 3*            100 /
   'WELL5' OPEN   LRAT    2913.68 669.71 303535.03 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   10224.8           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN    5899.63          350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN    6106.33          350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN    9932.23          350 6*
                                                                         RATE    /
/












/

DATES
01 SEP 2025 /
/

WCONHIST
-- wname   status control orate   wrate  grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    8861.01 933.34 918846.76 3*            100 /
   'WELL2' OPEN   LRAT    3008.29 696.79 312738.36 3*            100 /
   'WELL3' OPEN   LRAT    2931.47 676.15 304948.98 3*            100 /
   'WELL4' OPEN   LRAT    2825.47 649.12 294070.56 3*            100 /
   'WELL5' OPEN   LRAT    2910.19 669.87 302822.29 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   10168.79          350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN    5888.62          350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN    6098             350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN    9878.39          350 6*
                                                                         RATE    /
/












/

DATES
05 SEP 2025 /
/

WCONHIST
-- wname   status control orate   wrate  grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    8650.76 951.34 896967.52 3*            100 /
   'WELL2' OPEN   LRAT    2998.97 697.1  311464.46 3*            100 /
   'WELL3' OPEN   LRAT    2922.05 677.48 303654.87 3*            100 /
   'WELL4' OPEN   LRAT    2816.42 651.63 292820.24 3*            100 /
   'WELL5' OPEN   LRAT    2902.62 671.94 301706.56 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   10057.84          350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN    5858.92          350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN    6068.26          350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN    9771.8           350 6*
                                                                         RATE    /
/












/

DATES
12 SEP 2025 /
/

WCONHIST
-- wname   status control orate   wrate  grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    8392.32 976.61 870110.83 3*            100 /
   'WELL2' OPEN   LRAT    2993.55 700.26 310691.4  3*            100 /
   'WELL3' OPEN   LRAT    2916.7  680.8  302882.71 3*            100 /
   'WELL4' OPEN   LRAT    2811.85 655.4  292123.2  3*            100 /
   'WELL5' OPEN   LRAT    2898.86 675.57 301090.08 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   9958.27           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   5827.38           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   6035.02           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   9676.18           350 6*
                                                                         RATE    /
/












/

DATES
19 SEP 2025 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    7727.63 1059.14 801916.15 3*            100 /
   'WELL2' OPEN   LRAT    3000.83  705.7  311299.62 3*            100 /
   'WELL3' OPEN   LRAT    2924.16  685.82 303533.23 3*            100 /
   'WELL4' OPEN   LRAT    2820.5   660.8  292924.74 3*            100 /
   'WELL5' OPEN   LRAT    2907.85  680.87 301898.66 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   9830.74           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   5775.24           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   5980.17           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   9551.15           350 6*
                                                                         RATE    /
/












/

DATES
26 SEP 2025 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    6304.86 1394.68 653938.87 3*            100 /
   'WELL2' OPEN   LRAT    3040.84  716.11 315337.89 3*            100 /
   'WELL3' OPEN   LRAT    2964.3   695.83 307656.25 3*            100 /
   'WELL4' OPEN   LRAT    2860.44  670.73 297147.26 3*            100 /
   'WELL5' OPEN   LRAT    2947.98  690.85 306033.53 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   9639.12           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   5682.22           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   5886.13           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   9363.99           350 6*
                                                                         RATE    /
/












/

DATES
01 OCT 2025 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    5832.5  1586.93 605001.08 3*            100 /
   'WELL2' OPEN   LRAT    3063.38  721.7  317646.12 3*            100 /
   'WELL3' OPEN   LRAT    2986.76  701.22 309974.57 3*            100 /
   'WELL4' OPEN   LRAT    2882.55  676.18 299505.16 3*            100 /
   'WELL5' OPEN   LRAT    2970.19  696.2  308343.72 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   9549.72           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   5636.57           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   5839.67           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   9277.09           350 6*
                                                                         RATE    /
/












/

DATES
03 OCT 2025 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    4779.82 2198.75 496284.45 3*            100 /
   'WELL2' OPEN   LRAT    3131.23  740.55 324681.45 3*            100 /
   'WELL3' OPEN   LRAT    3053.92  720.69 316889.86 3*            100 /
   'WELL4' OPEN   LRAT    2949.3   696.02 306603.25 3*            100 /
   'WELL5' OPEN   LRAT    3037.05  715.97 315252.99 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   9274.55           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   5491.96           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   5691.69           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   9012.06           350 6*
                                                                         RATE    /
/



/

DATES
10 OCT 2025 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    4227.18 2634.85 438745.6  3*            100 /
   'WELL2' OPEN   LRAT    3157.25  755.42 327420.42 3*            100 /
   'WELL3' OPEN   LRAT    3077.79  736.46 319263.58 3*            100 /
   'WELL4' OPEN   LRAT    2973.78  711.39 309043.31 3*            100 /
   'WELL5' OPEN   LRAT    3061.14  731.75 317596.46 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   9096.44           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   5398.15           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   5594.22           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   8842.45           350 6*
                                                                         RATE    /
/
/

DATES
17 OCT 2025 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    3881.41 2976.26 402755.03 3*            100 /
   'WELL2' OPEN   LRAT    3152.64  766.99 326960.2  3*            100 /
   'WELL3' OPEN   LRAT    3070.22  748.08 318441.63 3*            100 /
   'WELL4' OPEN   LRAT    2965.69  723.55 307998.76 3*            100 /
   'WELL5' OPEN   LRAT    3053.1   743.15 316684.71 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   8977.4            350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   5336.41           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   5529.25           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   8729.49           350 6*
                                                                         RATE    /
/












/

DATES
24 OCT 2025 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    3620.85 3244.19 375613.8  3*            100 /
   'WELL2' OPEN   LRAT    3129.77  777.16 324593.87 3*            100 /
   'WELL3' OPEN   LRAT    3044.31  759.85 315750.34 3*            100 /
   'WELL4' OPEN   LRAT    2938.64  739.43 305029.13 3*            100 /
   'WELL5' OPEN   LRAT    3027.23  756.62 313988.6  3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   8886.03           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   5288.61           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   5479.46           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   8643.08           350 6*
                                                                         RATE    /
/












/

DATES
31 OCT 2025 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    3588.01 3279.53 372203.82 3*            100 /
   'WELL2' OPEN   LRAT    3126.03  778.67 324205.01 3*            100 /
   'WELL3' OPEN   LRAT    3040.33  761.9  315337.69 3*            100 /
   'WELL4' OPEN   LRAT    2934.17  741.88 304537.27 3*            100 /
   'WELL5' OPEN   LRAT    3023.08  758.86 313556.66 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   8874.26           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   5282.4            350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   5472.96           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   8631.95           350 6*
                                                                         RATE    /
/












/

DATES
01 NOV 2025 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    3424.62 3457.95 355229.84 3*            100 /
   'WELL2' OPEN   LRAT    3093.58  791.19 320849.22 3*            100 /
   'WELL3' OPEN   LRAT    3004.37  776.89 311612.79 3*            100 /
   'WELL4' OPEN   LRAT    2894.41  758.52 300327.26 3*            100 /
   'WELL5' OPEN   LRAT    2986.43  774.09 309753.3  3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   8807.45           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   5246.65           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   5435.93           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   8567.97           350 6*
                                                                         RATE    /
/












/

DATES
07 NOV 2025 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    3270.99 3648.08 339248.76 3*            100 /
   'WELL2' OPEN   LRAT    3044.24  809.68 315734.07 3*            100 /
   'WELL3' OPEN   LRAT    2950.41  798.54 306013.75 3*            100 /
   'WELL4' OPEN   LRAT    2836.13  784.57 294226.78 3*            100 /
   'WELL5' OPEN   LRAT    2932.03  796.59 304111.83 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   8732.84           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   5204.93           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   5393.26           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   8496.17           350 6*
                                                                         RATE    /
/












/

DATES
14 NOV 2025 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    3137.46 3826.34 325350.16 3*            100 /
   'WELL2' OPEN   LRAT    2985.7   834.52 309664.59 3*            100 /
   'WELL3' OPEN   LRAT    2886.45  826.51 299381.09 3*            100 /
   'WELL4' OPEN   LRAT    2763.53  815.66 286655.75 3*            100 /
   'WELL5' OPEN   LRAT    2865.57  824.63 297211.29 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   8659.2            350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   5161.85           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   5349.55           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   8423.74           350 6*
                                                                         RATE    /
/












/

DATES
21 NOV 2025 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    3019.55 3990.37 313118    3*            100 /
   'WELL2' OPEN   LRAT    2914.61  864.74 302281.72 3*            100 /
   'WELL3' OPEN   LRAT    2809.37  860.53 291378.66 3*            100 /
   'WELL4' OPEN   LRAT    2681.29  855.57 278112.09 3*            100 /
   'WELL5' OPEN   LRAT    2789.32  859.35 289301.47 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   8582.66           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   5115.39           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   5303.05           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   8348.71           350 6*
                                                                         RATE    /
/












/

DATES
28 NOV 2025 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    2971.21 4059.17 308103.86 3*            100 /
   'WELL2' OPEN   LRAT    2884.27  879.67 299132.32 3*            100 /
   'WELL3' OPEN   LRAT    2776.74  878.06 287992.15 3*            100 /
   'WELL4' OPEN   LRAT    2645.09  875.36 274353.08 3*            100 /
   'WELL5' OPEN   LRAT    2755.72  876.39 285812.35 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   8550.29           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   5095.41           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   5282.86           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   8316.81           350 6*
                                                                         RATE    /
/












/

DATES
01 DEC 2025 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    2908.54 4147.7  301604.86 3*            100 /
   'WELL2' OPEN   LRAT    2840.3   902.2  294576.53 3*            100 /
   'WELL3' OPEN   LRAT    2728.85  904.26 283033.45 3*            100 /
   'WELL4' OPEN   LRAT    2590.51  903.4  268662.94 3*            100 /
   'WELL5' OPEN   LRAT    2706.08  902.67 280660.3  3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   8506.37           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   5067.82           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   5255              350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   8273.09           350 6*
                                                                         RATE    /
/












/

DATES
05 DEC 2025 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    2805.58 4296.73 290928.07 3*            100 /
   'WELL2' OPEN   LRAT    2745.78  949.52 284769.1  3*            100 /
   'WELL3' OPEN   LRAT    2626.45  956.19 272402.71 3*            100 /
   'WELL4' OPEN   LRAT    2479.15  961.57 257088.12 3*            100 /
   'WELL5' OPEN   LRAT    2603.07  955.15 269978.76 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   8420.79           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   5012.59           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   5200.21           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   8187.5            350 6*
                                                                         RATE    /
/












/

DATES
12 DEC 2025 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    2713.4  4443.29 281369.1  3*            100 /
   'WELL2' OPEN   LRAT    2637.51 1006.91 273539.39 3*            100 /
   'WELL3' OPEN   LRAT    2508.75 1019.65 260192.46 3*            100 /
   'WELL4' OPEN   LRAT    2345.53 1032.24 243206.58 3*            100 /
   'WELL5' OPEN   LRAT    2479.16 1019.08 257110.97 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   8327.03           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4950.15           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   5138.33           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   8092.1            350 6*
                                                                         RATE    /
/












/

DATES
19 DEC 2025 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    2620.76 4589.32 271763.13 3*            100 /
   'WELL2' OPEN   LRAT    2505.04 1077.9  259778.71 3*            100 /
   'WELL3' OPEN   LRAT    2363.98 1097.75 245154.04 3*            100 /
   'WELL4' OPEN   LRAT    2170.27 1124.12 225035.93 3*            100 /
   'WELL5' OPEN   LRAT    2317.75 1102.16 240311.61 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   8218.95           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4875.9            350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   5065.11           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   7979.14           350 6*
                                                                         RATE    /
/












/

DATES
26 DEC 2025 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    2541.81 4716.64 263576.41 3*            100 /
   'WELL2' OPEN   LRAT    2356.35 1156.11 244281.94 3*            100 /
   'WELL3' OPEN   LRAT    2203.34 1185.22 228402.73 3*            100 /
   'WELL4' OPEN   LRAT    1983.79 1229.11 205577.52 3*            100 /
   'WELL5' OPEN   LRAT    2139.57 1196.57 221695.67 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   8109.15           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4800.08           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4989.47           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   7864.51           350 6*
                                                                         RATE    /
/












/

DATES
01 JAN 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    2528.48 4738.06 262193.69 3*            100 /
   'WELL2' OPEN   LRAT    2330.73 1169.86 241584.55 3*            100 /
   'WELL3' OPEN   LRAT    2175.77 1200.72 225501.52 3*            100 /
   'WELL4' OPEN   LRAT    1951.85 1248.58 202217.17 3*            100 /
   'WELL5' OPEN   LRAT    2109    1214.5  218485.91 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   8090.43           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4787.16           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4976.56           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   7844.99           350 6*
                                                                         RATE    /
/












/

DATES
02 JAN 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    2447.73 4893.28 253819.92 3*            100 /
   'WELL2' OPEN   LRAT    2103.59 1298.95 217912.49 3*            100 /
   'WELL3' OPEN   LRAT    1938.9  1340.39 200812.99 3*            100 /
   'WELL4' OPEN   LRAT    1724.98 1400.09 178608.74 3*            100 /
   'WELL5' OPEN   LRAT    1884.07 1358.88 195097.72 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7936.04           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4683.81           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4871.88           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   7694.57           350 6*
                                                                         RATE    /
/












/

DATES
09 JAN 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    2368.43 5045.86 245596.77 3*            100 /
   'WELL2' OPEN   LRAT    1875.83 1450.37 194233.38 3*            100 /
   'WELL3' OPEN   LRAT    1708.97 1497.87 176918.33 3*            100 /
   'WELL4' OPEN   LRAT    1525.92 1551.16 157974.07 3*            100 /
   'WELL5' OPEN   LRAT    1684.4  1506.51 174393.8  3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7775.68           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4579.03           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4764.29           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   7546.42           350 6*
                                                                         RATE    /
/












/

DATES
16 JAN 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    2287.38 5188.39 237192.97 3*            100 /
   'WELL2' OPEN   LRAT    1674.4  1600.68 173367.98 3*            100 /
   'WELL3' OPEN   LRAT    1511.42 1652.2  156479.35 3*            100 /
   'WELL4' OPEN   LRAT    1361.94 1689.79 141033.46 3*            100 /
   'WELL5' OPEN   LRAT    1515.59 1644.03 156942.32 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7626.08           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4482.04           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4663              350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   7411.51           350 6*
                                                                         RATE    /
/












/

DATES
23 JAN 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    2208.86 5317.34 229050.34 3*            100 /
   'WELL2' OPEN   LRAT    1515.86 1740.81 157223.86 3*            100 /
   'WELL3' OPEN   LRAT    1347.88 1791.6  139617.65 3*            100 /
   'WELL4' OPEN   LRAT    1222.71 1813.12 126655.53 3*            100 /
   'WELL5' OPEN   LRAT    1382.04 1768.45 143344.56 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7498.68           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4397.24           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4576.36           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   7296.55           350 6*
                                                                         RATE    /
/












/

DATES
30 JAN 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    2186.41 5352.76 226722.5  3*            100 /
   'WELL2' OPEN   LRAT    1474.99 1780.29 153122.1  3*            100 /
   'WELL3' OPEN   LRAT    1305.47 1830.73 135331.2  3*            100 /
   'WELL4' OPEN   LRAT    1185.26 1847.41 122793.25 3*            100 /
   'WELL5' OPEN   LRAT    1346.33 1802.97 139723.11 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7464.5            350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4374.28           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4553.04           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   7265.53           350 6*
                                                                         RATE    /
/












/

DATES
01 FEB 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    2129.75 5436.97 220847.28 3*            100 /
   'WELL2' OPEN   LRAT    1380.75 1874.37 143450.05 3*            100 /
   'WELL3' OPEN   LRAT    1209.79 1924.01 125576.67 3*            100 /
   'WELL4' OPEN   LRAT    1098.01 1930.98 113893.1  3*            100 /
   'WELL5' OPEN   LRAT    1263.05 1886.4  131181.38 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7389.3            350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4322.6            350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4500.92           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   7196.03           350 6*
                                                                         RATE    /
/












/

DATES
06 FEB 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    2048.54 5540.44 212425.68 3*            100 /
   'WELL2' OPEN   LRAT    1260.57 1990.35 130954.97 3*            100 /
   'WELL3' OPEN   LRAT    1088.93 2041.5  113066.38 3*            100 /
   'WELL4' OPEN   LRAT     988.13 2037.5  102504.63 3*            100 /
   'WELL5' OPEN   LRAT    1154.12 1991.81 119830.77 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7306.32           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4263.99           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4441.01           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   7117.69           350 6*
                                                                         RATE    /
/












/

DATES
13 FEB 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    1969.01 5639.83 204178.52 3*            100 /
   'WELL2' OPEN   LRAT    1145.03 2106.19 118927.38 3*            100 /
   'WELL3' OPEN   LRAT     970    2161.67 100738.68 3*            100 /
   'WELL4' OPEN   LRAT     880.46 2147.49  91328.12 3*            100 /
   'WELL5' OPEN   LRAT    1047.83 2097.17 108770.44 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7232.93           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4211.91           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4387.15           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   7047.94           350 6*
                                                                         RATE    /
/












/

DATES
20 FEB 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    1890.74 5735.06 196062.74 3*            100 /
   'WELL2' OPEN   LRAT    1029.19 2220.23 106892.58 3*            100 /
   'WELL3' OPEN   LRAT     856.88 2284.65  89006.86 3*            100 /
   'WELL4' OPEN   LRAT     777.24 2260.94  80697.07 3*            100 /
   'WELL5' OPEN   LRAT     941.64 2203.7   97741.34 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7170.39           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4168.16           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4339.88           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6988.58           350 6*
                                                                         RATE    /
/












/

DATES
27 FEB 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    1868.4  5761.64 193745.79 3*            100 /
   'WELL2' OPEN   LRAT     996.22 2253.35 103471.77 3*            100 /
   'WELL3' OPEN   LRAT     826.18 2319.47  85839.83 3*            100 /
   'WELL4' OPEN   LRAT     749.71 2292.83  77865.75 3*            100 /
   'WELL5' OPEN   LRAT     911.27 2234.98  94583.52 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7154.02           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4157              350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4327.24           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6973.15           350 6*
                                                                         RATE    /
/












/

DATES
01 MAR 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    1812.67 5828.78 187966.87 3*            100 /
   'WELL2' OPEN   LRAT     915.79 2337.38  95105.48 3*            100 /
   'WELL3' OPEN   LRAT     759.37 2400.68  78914.63 3*            100 /
   'WELL4' OPEN   LRAT     691.25 2366.46  71808    3*            100 /
   'WELL5' OPEN   LRAT     836.9  2314.19  86877.93 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7117.92           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4133.14           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4298.57           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6939.37           350 6*
                                                                         RATE    /
/












/

DATES
06 MAR 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    1731.69 5921.78 179569.92 3*            100 /
   'WELL2' OPEN   LRAT     814.67 2450.71  84611.41 3*            100 /
   'WELL3' OPEN   LRAT     685.76 2497.47  71276.62 3*            100 /
   'WELL4' OPEN   LRAT     628.13 2453.63  65286.37 3*            100 /
   'WELL5' OPEN   LRAT     745.92 2418.99  77453.22 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7077.47           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4107.32           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4265.23           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6902.05           350 6*
                                                                         RATE    /
/












/

DATES
13 MAR 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    1650.24 6012.98 171124.01 3*            100 /
   'WELL2' OPEN   LRAT     731.99 2551.77  76052.77 3*            100 /
   'WELL3' OPEN   LRAT     629.55 2579.2   65409.9  3*            100 /
   'WELL4' OPEN   LRAT     578.66 2527.3   60098.15 3*            100 /
   'WELL5' OPEN   LRAT     673.33 2511.31  69944.09 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7047.55           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4088.48           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4239.66           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6874.68           350 6*
                                                                         RATE    /
/












/

DATES
20 MAR 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    1570.84 6106.25 162890.41 3*            100 /
   'WELL2' OPEN   LRAT     667.95 2637.86  69403.33 3*            100 /
   'WELL3' OPEN   LRAT     583.3  2647.9   60587.83 3*            100 /
   'WELL4' OPEN   LRAT     537.15 2589.79  55784.47 3*            100 /
   'WELL5' OPEN   LRAT     617.04 2590.12  64072.08 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7027.55           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4075.27           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4221.86           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6856.6            350 6*
                                                                         RATE    /
/












/

DATES
27 MAR 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    1513.42 6172.36 156936.02 3*            100 /
   'WELL2' OPEN   LRAT     629.9  2692.31  65430.28 3*            100 /
   'WELL3' OPEN   LRAT     554.04 2691.42  57541.71 3*            100 /
   'WELL4' OPEN   LRAT     510.99 2629.72  53065.19 3*            100 /
   'WELL5' OPEN   LRAT     582.81 2639.25  60512.68 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7017.33           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4068.07           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4212.45           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6847.34           350 6*
                                                                         RATE    /
/












/

DATES
01 APR 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    1490.51 6198.33 154560.56 3*            100 /
   'WELL2' OPEN   LRAT     615.59 2712.82  63939.88 3*            100 /
   'WELL3' OPEN   LRAT     543.01 2708.05  56392.94 3*            100 /
   'WELL4' OPEN   LRAT     501.14 2645.01  52040.32 3*            100 /
   'WELL5' OPEN   LRAT     569.96 2657.74  59178.42 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7013.83           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4065.53           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4209.16           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6844.17           350 6*
                                                                         RATE    /
/












/

DATES
03 APR 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    1411.04 6291.29 146319.57 3*            100 /
   'WELL2' OPEN   LRAT     573.41 2773.92  59543.5  3*            100 /
   'WELL3' OPEN   LRAT     510.12 2759.38  52962.42 3*            100 /
   'WELL4' OPEN   LRAT     471.11 2692.59  48910.23 3*            100 /
   'WELL5' OPEN   LRAT     531.86 2713.5   55216.46 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7006.6            350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4059.64           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4201.37           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6837.62           350 6*
                                                                         RATE    /
/












/

DATES
10 APR 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    1333.07 6384.38 138234.75 3*            100 /
   'WELL2' OPEN   LRAT     537.65 2827.28  55810.26 3*            100 /
   'WELL3' OPEN   LRAT     481.42 2804.99  49972.85 3*            100 /
   'WELL4' OPEN   LRAT     444.73 2735.1   46161.4  3*            100 /
   'WELL5' OPEN   LRAT     499.61 2762.73  51856.85 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7003.04           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4056.05           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4196.36           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6834.63           350 6*
                                                                         RATE    /
/












/

DATES
17 APR 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    1257.62 6478.47 130410.79 3*            100 /
   'WELL2' OPEN   LRAT     506.79 2874.32  52597.44 3*            100 /
   'WELL3' OPEN   LRAT     455.86 2845.98  47315.88 3*            100 /
   'WELL4' OPEN   LRAT     421.38 2773.33  43733.62 3*            100 /
   'WELL5' OPEN   LRAT     471.63 2806.55  48947.31 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7002.39           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4054.38           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4193.53           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6834.44           350 6*
                                                                         RATE    /
/












/

DATES
24 APR 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    1185.23 6572.59 122903.71 3*            100 /
   'WELL2' OPEN   LRAT     479.79 2916.77  49794.39 3*            100 /
   'WELL3' OPEN   LRAT     433.05 2883.33  44946.07 3*            100 /
   'WELL4' OPEN   LRAT     400.84 2808.64  41598.6  3*            100 /
   'WELL5' OPEN   LRAT     446.75 2846.19  46364.32 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7004.2            350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4054.42           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4192.52           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6836.7            350 6*
                                                                         RATE    /
/












/

DATES
01 MAY 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    1116.63 6666.48 115790.15 3*            100 /
   'WELL2' OPEN   LRAT     455.4  2955.69  47264.42 3*            100 /
   'WELL3' OPEN   LRAT     412.45 2918.07  42798.88 3*            100 /
   'WELL4' OPEN   LRAT     382.34 2841.31  39671.05 3*            100 /
   'WELL5' OPEN   LRAT     424.44 2882.4   44047.91 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7008.23           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4055.99           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4193.08           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6841.12           350 6*
                                                                         RATE    /
/












/

DATES
08 MAY 2026 /
/

WCONHIST
-- wname   status control orate   wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    1051.76 6758.07 109063.89 3*            100 /
   'WELL2' OPEN   LRAT     433.21 2991.58  44961.14 3*            100 /
   'WELL3' OPEN   LRAT     393.51 2950.46  40819.99 3*            100 /
   'WELL4' OPEN   LRAT     365.48 2871.66  37915.09 3*            100 /
   'WELL5' OPEN   LRAT     404.54 2915.82  41974.58 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7013.93           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4058.77           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4194.97           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6847.27           350 6*
                                                                         RATE    /
/












/

DATES
15 MAY 2026 /
/

WCONHIST
-- wname   status control orate  wrate   grate     VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    991.82 6846.8  102848.2  3*            100 /
   'WELL2' OPEN   LRAT    413.17 3024.99  42873.5  3*            100 /
   'WELL3' OPEN   LRAT    376.05 2980.64  38994.7  3*            100 /
   'WELL4' OPEN   LRAT    350.09 2899.74  36307.16 3*            100 /
   'WELL5' OPEN   LRAT    386.54 2946.96  40091.64 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7021.19           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4062.59           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4198.02           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6854.89           350 6*
                                                                         RATE    /
/












/

DATES
22 MAY 2026 /
/

WCONHIST
-- wname   status control orate  wrate   grate    VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    937.25 6933.83 97189.3  3*            100 /
   'WELL2' OPEN   LRAT    394.84 3056.4  40954.38 3*            100 /
   'WELL3' OPEN   LRAT    359.95 3008.6  37311.63 3*            100 /
   'WELL4' OPEN   LRAT    335.77 2925.62 34816.12 3*            100 /
   'WELL5' OPEN   LRAT    370.01 2975.83 38360.99 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7030.05           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4067.32           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4202.19           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6863.98           350 6*
                                                                         RATE    /
/












/

DATES
29 MAY 2026 /
/

WCONHIST
-- wname   status control orate  wrate   grate    VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    914.96 6970.41 94877.47 3*            100 /
   'WELL2' OPEN   LRAT    387.32 3069.34 40165.73 3*            100 /
   'WELL3' OPEN   LRAT    353.35 3020.01 36624.28 3*            100 /
   'WELL4' OPEN   LRAT    329.83 2936.29 34200.34 3*            100 /
   'WELL5' OPEN   LRAT    363.14 2987.64 37642.96 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7034.1            350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4069.5            350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4204.18           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6868.11           350 6*
                                                                         RATE    /
/












/

DATES
01 JUN 2026 /
/

WCONHIST
-- wname   status control orate  wrate   grate    VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    886.78 7018.49 91955.54 3*            100 /
   'WELL2' OPEN   LRAT    377.65 3085.9  39151.35 3*            100 /
   'WELL3' OPEN   LRAT    344.94 3034.62 35749    3*            100 /
   'WELL4' OPEN   LRAT    322.25 2949.98 33411.05 3*            100 /
   'WELL5' OPEN   LRAT    354.38 3002.71 36728.32 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7039.92           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4072.64           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4207.13           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6873.98           350 6*
                                                                         RATE    /
/












/

DATES
05 JUN 2026 /
/

WCONHIST
-- wname   status control orate  wrate   grate    VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    841.5  7098.96 87260.49 3*            100 /
   'WELL2' OPEN   LRAT    361.85 3112.48 37501.97 3*            100 /
   'WELL3' OPEN   LRAT    331.18 3058.41 34321.12 3*            100 /
   'WELL4' OPEN   LRAT    309.83 2972.22 32119.14 3*            100 /
   'WELL5' OPEN   LRAT    340.19 3027.04 35253.82 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7050.86           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4078.63           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4212.83           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6885.01           350 6*
                                                                         RATE    /
/












/

DATES
12 JUN 2026 /
/

WCONHIST
-- wname   status control orate  wrate   grate    VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    800.11 7174.58 82968.24 3*            100 /
   'WELL2' OPEN   LRAT    347.28 3136.73 35984.25 3*            100 /
   'WELL3' OPEN   LRAT    318.45 3080.56 33001.77 3*            100 /
   'WELL4' OPEN   LRAT    298.31 2992.95 30924.46 3*            100 /
   'WELL5' OPEN   LRAT    327.09 3049.66 33895.41 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7062.13           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4084.93           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4218.84           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6896.38           350 6*
                                                                         RATE    /
/












/

DATES
19 JUN 2026 /
/

WCONHIST
-- wname   status control orate  wrate   grate    VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    762.51 7247.38 79069.23 3*            100 /
   'WELL2' OPEN   LRAT    333.82 3159.22 34584.9  3*            100 /
   'WELL3' OPEN   LRAT    306.65 3101.22 31777.3  3*            100 /
   'WELL4' OPEN   LRAT    287.6  3012.36 29813.75 3*            100 /
   'WELL5' OPEN   LRAT    314.92 3070.84 32634.82 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7073.78           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4091.51           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4225.12           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6908.17           350 6*
                                                                         RATE    /
/












/

DATES
26 JUN 2026 /
/

WCONHIST
-- wname   status control orate  wrate   grate    VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    737.62 7298.98 76488.01 3*            100 /
   'WELL2' OPEN   LRAT    324.77 3174.44 33645.39 3*            100 /
   'WELL3' OPEN   LRAT    298.69 3115.26 30954.22 3*            100 /
   'WELL4' OPEN   LRAT    280.36 3025.52 29061.34 3*            100 /
   'WELL5' OPEN   LRAT    306.72 3085.24 31786.29 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7082.42           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4096.4            350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4229.82           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6916.9            350 6*
                                                                         RATE    /
/












/

DATES
01 JUL 2026 /
/

WCONHIST
-- wname   status control orate  wrate   grate    VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    728.02 7319.5  75492.92 3*            100 /
   'WELL2' OPEN   LRAT    321.23 3180.36 33278.7  3*            100 /
   'WELL3' OPEN   LRAT    295.58 3120.73 30631.8  3*            100 /
   'WELL4' OPEN   LRAT    277.54 3030.64 28767.42 3*            100 /
   'WELL5' OPEN   LRAT    303.51 3090.85 31454.62 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7085.94           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4098.39           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4231.73           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6920.45           350 6*
                                                                         RATE    /
/












/

DATES
03 JUL 2026 /
/

WCONHIST
-- wname   status control orate  wrate   grate    VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    696.76 7387.97 72250.89 3*            100 /
   'WELL2' OPEN   LRAT    309.71 3199.92 32083.32 3*            100 /
   'WELL3' OPEN   LRAT    285.38 3138.84 29572.9  3*            100 /
   'WELL4' OPEN   LRAT    268.21 3047.8  27802.22 3*            100 /
   'WELL5' OPEN   LRAT    293.03 3109.4  30366.78 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7098.61           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4105.59           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4238.71           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6933.23           350 6*
                                                                         RATE    /
/












/

DATES
10 JUL 2026 /
/

WCONHIST
-- wname   status control orate  wrate   grate    VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    667.8  7451.7  69248.1  3*            100 /
   'WELL2' OPEN   LRAT    299.03 3218.34 30975.75 3*            100 /
   'WELL3' OPEN   LRAT    275.87 3155.83 28586.83 3*            100 /
   'WELL4' OPEN   LRAT    259.46 3064.06 26895.59 3*            100 /
   'WELL5' OPEN   LRAT    283.24 3126.82 29350.91 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7111.23           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4112.85           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4245.78           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6945.91           350 6*
                                                                         RATE    /
/












/

DATES
17 JUL 2026 /
/

WCONHIST
-- wname   status control orate  wrate   grate    VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    641.13 7511.71 66482.65 3*            100 /
   'WELL2' OPEN   LRAT    289.11 3235.68 29946.88 3*            100 /
   'WELL3' OPEN   LRAT    267.08 3172.11 27677.19 3*            100 /
   'WELL4' OPEN   LRAT    251.33 3079.66 26052.63 3*            100 /
   'WELL5' OPEN   LRAT    274.06 3143.42 28402.42 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7123.81           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4120.15           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4252.88           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6958.53           350 6*
                                                                         RATE    /
/












/

DATES
24 JUL 2026 /
/

WCONHIST
-- wname   status control orate  wrate   grate    VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    616.48 7568.9  63927.05 3*            100 /
   'WELL2' OPEN   LRAT    279.86 3252.06 28988.69 3*            100 /
   'WELL3' OPEN   LRAT    258.9  3187.49 26828.97 3*            100 /
   'WELL4' OPEN   LRAT    243.68 3094.66 25260.84 3*            100 /
   'WELL5' OPEN   LRAT    265.45 3159.28 27509.59 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7136.32           350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4127.46           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4260.01           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6971.1            350 6*
                                                                         RATE    /
/












/

DATES
31 JUL 2026 /
/

WCONHIST
-- wname   status control orate  wrate   grate    VFP_n ALQ THP BHP wet-gas_rate NGL_rate
   'WELL1' OPEN   LRAT    613.01 7576.76 63566.45 3*            100 /
   'WELL2' OPEN   LRAT    278.56 3254.29 28854.43 3*            100 /
   'WELL3' OPEN   LRAT    257.74 3189.58 26708.85 3*            100 /
   'WELL4' OPEN   LRAT    242.61 3096.72 25148.99 3*            100 /
   'WELL5' OPEN   LRAT    264.24 3161.46 27383.78 3*            100 /
/

WCONINJH
-- wname   injector_type status observed_inj.rate BHP THP VFP_n oil_concentration sc_oil_proport sc_water_proport sc_gas_proport control
   'WELL6' WATER         OPEN   7138.1            350 6*
                                                                         RATE    /
   'WELL7' WATER         OPEN   4128.51           350 6*
                                                                         RATE    /
   'WELL8' WATER         OPEN   4261.03           350 6*
                                                                         RATE    /
   'WELL9' WATER         OPEN   6972.89           350 6*
                                                                         RATE    /
/












/

DATES
01 AUG 2026 /
/
