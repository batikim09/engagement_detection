#sanity check
python ./feat_prepare_by_functionals.py -input ./meta/meta.engagement.deepcp.indiv.temp.csv -mt 1:2:3:4:5:6:7 -f_idx 9 -out ../../features/DEEPCP.EMO.func.temp.indiv -base "/Users/kimj/SURFdrive/SQL/DEEP_CV_POSE/"

#emotion feature by functinals (baseline e.g. svmrank)
python ./feat_prepare_by_functionals.py -input ./meta/meta.engagement.deepcp.indiv.nan.multiview.csv -mt 1:2:3:4:5:6:7 -f_idx 9 -out ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan -base "/Users/kimj/SURFdrive/SQL/DEEP_CV_POSE/" --nan -n_cc 8 -c_idx 2 -w_idx 3
python ./feat_prepare_by_functionals.py -input ./meta/meta.engagement.deepcp.indiv.nan.multiview.csv -mt 1:2:3:4:5:6:7 -f_idx 8 -out ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan -base "/Users/kimj/SURFdrive/SQL/DEEP_CV_POSE/" --nan -n_cc 8 -c_idx 2 -w_idx 3
python ./feat_prepare_by_functionals.py -input ./meta/meta.engagement.deepcp.indiv.nan.multiview.csv -mt 1:2:3:4:5:6:7 -f_idx 10 -out ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.OPENCV.func.indiv.nan -base "/Users/kimj/SURFdrive/SQL/DEEP_CV_POSE/" --nan -n_cc 8 -c_idx 2 -w_idx 3
python ./feat_prepare_by_functionals.py -input ./meta/meta.engagement.deepcp.indiv.nan.multiview.csv -mt 1:2:3:4:5:6:7 -f_idx 11 -out ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.NV.func.indiv.nan -base "/Users/kimj/SURFdrive/SQL/DEEP_CV_POSE/" --nan -n_cc 8 -c_idx 2 -w_idx 3

python ./feat_prepare_by_functionals.py -input ./meta/meta.engagement.deepcp.indiv.nan.multiview.csv -mt 1:2:3:4:5:6:7 -f_idx 8 -out ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta -base "/Users/kimj/SURFdrive/SQL/DEEP_CV_POSE/" --nan --delta -n_cc 8 -c_idx 2 -w_idx 3
python ./feat_prepare_by_functionals.py -input ./meta/meta.engagement.deepcp.indiv.nan.multiview.csv -mt 1:2:3:4:5:6:7 -f_idx 8 -out ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.abs -base "/Users/kimj/SURFdrive/SQL/DEEP_CV_POSE/" --nan --delta --absolute -n_cc 8 -c_idx 2 -w_idx 3

python ../keras_ser/feat_prepare_for_LSTM.py -input ./meta/meta.engagement.deepcp.paired.csv -m_steps 125 -c_len 1 --two_d -c_idx 2 -mt 1:4:5:6 -f_idx 7 -n_cc 8 -out ../../features/engagement/DEEPCP.125.2d.paired -base "/Users/kimj/SURFdrive/SQL/DEEP_CV_POSE/"

python ../keras_ser/feat_prepare_for_LSTM.py -input ./meta/meta.engagement.deepcp.paired.diff.csv -m_steps 125 -c_len 1 --two_d -c_idx 2 -mt 1:4:5:6 -f_idx 7 -n_cc 8 -out ../../features/engagement/DEEPCP.125.2d.paired.diff -base "/Users/kimj/SURFdrive/SQL/DEEP_CV_POSE/"

python ../keras_ser/feat_prepare_for_LSTM.py -input ./meta/meta.engagement.deepcp.indiv.csv -m_steps 125 -c_len 1 --two_d -c_idx 2 -mt 1:4:5:6 -f_idx 7 -n_cc 8 -out ../../features/engagement/DEEPCP.125.2d.indiv -base "/Users/kimj/SURFdrive/SQL/DEEP_CV_POSE/"