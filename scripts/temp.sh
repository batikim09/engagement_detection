echo "ENGAGEMENT MANUAL -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 128 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -cs 128 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -kf 10 --unweighted

echo "ENGAGEMENT MANUAL set 0 -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 128 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -cs 128 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 0 -valid_idx 1

echo "ENGAGEMENT MANUAL set 1 -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 128 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -cs 128 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 1 -valid_idx 2

echo "ENGAGEMENT MANUAL set 2 -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 128 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -cs 128 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 2 -valid_idx 3

echo "ENGAGEMENT MANUAL set 3 -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 128 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -cs 128 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 3 -valid_idx 4

echo "ENGAGEMENT MANUAL set 4 -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 128 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -cs 128 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 4 -valid_idx 5

echo "ENGAGEMENT MANUAL set 5 -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 128 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -cs 128 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 5 -valid_idx 6

echo "ENGAGEMENT MANUAL set 6 -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 128 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -cs 128 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 6 -valid_idx 7

echo "ENGAGEMENT MANUAL set 7 -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 128 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -cs 128 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 7 -valid_idx 0

#DNN
echo "ENGAGEMENT MANUAL set 0 -f_dnn_depth 3 --f_dnn --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 256 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -f_dnn_depth 3 --f_dnn --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 256 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 0 -valid_idx 1

echo "ENGAGEMENT MANUAL set 1 -f_dnn_depth 3 --f_dnn --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 256 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -f_dnn_depth 3 --f_dnn --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 256 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 1 -valid_idx 2

echo "ENGAGEMENT MANUAL set 2 -f_dnn_depth 3 --f_dnn --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 256 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -f_dnn_depth 3 --f_dnn --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 256 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 2 -valid_idx 3

echo "ENGAGEMENT MANUAL set 3 -f_dnn_depth 3 --f_dnn --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 256 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -f_dnn_depth 3 --f_dnn --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 256 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 3 -valid_idx 4

echo "ENGAGEMENT MANUAL set 4 -f_dnn_depth 3 --f_dnn --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 256 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -f_dnn_depth 3 --f_dnn --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 256 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 4 -valid_idx 5

echo "ENGAGEMENT MANUAL set 5 -f_dnn_depth 3 --f_dnn --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 256 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -f_dnn_depth 3 --f_dnn --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 256 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 5 -valid_idx 6

echo "ENGAGEMENT MANUAL set 6 -f_dnn_depth 3 --f_dnn --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 256 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -f_dnn_depth 3 --f_dnn --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 256 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 6 -valid_idx 7

echo "ENGAGEMENT MANUAL set 7 -f_dnn_depth 3 --f_dnn --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 256 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -f_dnn_depth 3 --f_dnn --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 256 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 7 -valid_idx 0



#paired diff features
echo "ENGAGEMENT PAIRED_DIFF MANUAL set 0 -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 128 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -cs 128 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.diff.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 0 -valid_idx 1

echo "ENGAGEMENT PAIRED_DIFF MANUAL set 1 -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 128 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -cs 128 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.diff.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 1 -valid_idx 2

echo "ENGAGEMENT PAIRED_DIFF MANUAL set 2 -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 128 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -cs 128 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.diff.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 2 -valid_idx 3

echo "ENGAGEMENT PAIRED_DIFF MANUAL set 3 -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 128 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -cs 128 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.diff.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 3 -valid_idx 4

echo "ENGAGEMENT PAIRED_DIFF MANUAL set 4 -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 128 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -cs 128 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.diff.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 4 -valid_idx 5

echo "ENGAGEMENT PAIRED_DIFF MANUAL set 5 -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 128 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -cs 128 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.diff.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 5 -valid_idx 6

echo "ENGAGEMENT PAIRED_DIFF MANUAL set 6 -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 128 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -cs 128 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.diff.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 6 -valid_idx 7

echo "ENGAGEMENT PAIRED_DIFF MANUAL set 7 -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -nn 128 -l2 0.0001"	>> ./output/engagement.DEEPCP.txt
python ../keras_ser/dnn.framelevel.elm.py -dnn_depth 3 --f_lstm --post_elm -r_valid 0.1 -b 128 -e 20 -p 5 -cs 128 -l2 0.001 -t_max 125 -dt ../../features/engagement/DEEPCP.125.2d.paired.diff.h5 -mt ordinal:2:0:: -log ./output/temp.log -ot ./output/engagement.DEEPCP.txt -test_idx 7 -valid_idx 0