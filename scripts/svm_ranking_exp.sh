#sanity checks
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan.h5 -test_idx 0 -valid_idx 1 -train_idx 2 -c 1.0 -log './output/sanity.txt' -error './output/sanity'
python ./ranking_dnn_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan.h5 -test_idx 0 -valid_idx 1 -train_idx 2 -c 1.0 -log './output/sanity.txt' -error './output/sanity'

#svm ranking exp using emotion features
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan.h5 -test_idx 0 -valid_idx 1 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.txt' -error './output/svmrank.emo'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan.h5 -test_idx 1 -valid_idx 2 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.txt' -error './output/svmrank.emo'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan.h5 -test_idx 2 -valid_idx 3 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.txt' -error './output/svmrank.emo'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan.h5 -test_idx 3 -valid_idx 4 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.txt' -error './output/svmrank.emo'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan.h5 -test_idx 4 -valid_idx 5 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.txt' -error './output/svmrank.emo'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan.h5 -test_idx 5 -valid_idx 6 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.txt' -error './output/svmrank.emo'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan.h5 -test_idx 6 -valid_idx 7 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.txt' -error './output/svmrank.emo'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan.h5 -test_idx 7 -valid_idx 0 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.txt' -error './output/svmrank.emo'

#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan.min.h5 -test_idx 0 -valid_idx 1 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.min.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan.min.h5 -test_idx 1 -valid_idx 2 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.min.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan.min.h5 -test_idx 2 -valid_idx 3 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.min.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan.min.h5 -test_idx 3 -valid_idx 4 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.min.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan.min.h5 -test_idx 4 -valid_idx 5 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.min.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan.min.h5 -test_idx 5 -valid_idx 6 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.min.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan.min.h5 -test_idx 6 -valid_idx 7 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.min.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan.min.h5 -test_idx 7 -valid_idx 0 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.min.txt'

#svm ranking exp using open cv features
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.OPENCV.func.indiv.nan.h5 -test_idx 0 -valid_idx 1 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.opencv.txt' -error './output/svmrank.opencv'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.OPENCV.func.indiv.nan.h5 -test_idx 1 -valid_idx 2 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.opencv.txt' -error './output/svmrank.opencv'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.OPENCV.func.indiv.nan.h5 -test_idx 2 -valid_idx 3 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.opencv.txt' -error './output/svmrank.opencv'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.OPENCV.func.indiv.nan.h5 -test_idx 3 -valid_idx 4 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.opencv.txt' -error './output/svmrank.opencv'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.OPENCV.func.indiv.nan.h5 -test_idx 4 -valid_idx 5 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.opencv.txt' -error './output/svmrank.opencv'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.OPENCV.func.indiv.nan.h5 -test_idx 5 -valid_idx 6 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.opencv.txt' -error './output/svmrank.opencv'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.OPENCV.func.indiv.nan.h5 -test_idx 6 -valid_idx 7 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.opencv.txt' -error './output/svmrank.opencv'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.OPENCV.func.indiv.nan.h5 -test_idx 7 -valid_idx 0 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.opencv.txt' -error './output/svmrank.opencv'

#svm ranking exp using nv turn-taking features
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.NV.func.indiv.nan.h5 -test_idx 0 -valid_idx 1 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.nv.txt' -error './output/svmrank.nv'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.NV.func.indiv.nan.h5 -test_idx 1 -valid_idx 2 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.nv.txt' -error './output/svmrank.nv'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.NV.func.indiv.nan.h5 -test_idx 2 -valid_idx 3 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.nv.txt' -error './output/svmrank.nv'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.NV.func.indiv.nan.h5 -test_idx 3 -valid_idx 4 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.nv.txt' -error './output/svmrank.nv'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.NV.func.indiv.nan.h5 -test_idx 4 -valid_idx 5 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.nv.txt' -error './output/svmrank.nv'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.NV.func.indiv.nan.h5 -test_idx 5 -valid_idx 6 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.nv.txt' -error './output/svmrank.nv'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.NV.func.indiv.nan.h5 -test_idx 6 -valid_idx 7 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.nv.txt' -error './output/svmrank.nv'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.NV.func.indiv.nan.h5 -test_idx 7 -valid_idx 0 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.nv.txt' -error './output/svmrank.nv'

#svm ranking exp using pose features
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.h5 -test_idx 0 -valid_idx 1 -c 0.05,0.1,0.5,1.0,5.0,10.0 -log './output/svmrank.pose.txt' -error './output/svmrank.pose'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.h5 -test_idx 1 -valid_idx 2 -c 0.05,0.1,0.5,1.0,5.0,10.0 -log './output/svmrank.pose.txt' -error './output/svmrank.pose'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.h5 -test_idx 2 -valid_idx 3 -c 0.05,0.1,0.5,1.0,5.0,10.0 -log './output/svmrank.pose.txt' -error './output/svmrank.pose'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.h5 -test_idx 3 -valid_idx 4 -c 0.05,0.1,0.5,1.0,5.0,10.0 -log './output/svmrank.pose.txt' -error './output/svmrank.pose'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.h5 -test_idx 4 -valid_idx 5 -c 0.05,0.1,0.5,1.0,5.0,10.0 -log './output/svmrank.pose.txt' -error './output/svmrank.pose'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.h5 -test_idx 5 -valid_idx 6 -c 0.05,0.1,0.5,1.0,5.0,10.0 -log './output/svmrank.pose.txt' -error './output/svmrank.pose'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.h5 -test_idx 6 -valid_idx 7 -c 0.05,0.1,0.5,1.0,5.0,10.0 -log './output/svmrank.pose.txt' -error './output/svmrank.pose'
python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.h5 -test_idx 7 -valid_idx 0 -c 0.05,0.1,0.5,1.0,5.0,10.0 -log './output/svmrank.pose.txt' -error './output/svmrank.pose'

#delta features
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.h5 -test_idx 0 -valid_idx 1 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.pose.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.h5 -test_idx 1 -valid_idx 2 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.pose.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.h5 -test_idx 2 -valid_idx 3 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.pose.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.h5 -test_idx 3 -valid_idx 4 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.pose.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.h5 -test_idx 4 -valid_idx 5 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.pose.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.h5 -test_idx 5 -valid_idx 6 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.pose.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.h5 -test_idx 6 -valid_idx 7 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.pose.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.h5 -test_idx 7 -valid_idx 0 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.pose.txt'

#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.h5 -test_idx 0 -valid_idx 1 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 30 -log './output/svmrank.pose.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.h5 -test_idx 1 -valid_idx 2 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 30 -log './output/svmrank.pose.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.h5 -test_idx 2 -valid_idx 3 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 30 -log './output/svmrank.pose.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.h5 -test_idx 3 -valid_idx 4 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 30 -log './output/svmrank.pose.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.h5 -test_idx 4 -valid_idx 5 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 30 -log './output/svmrank.pose.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.h5 -test_idx 5 -valid_idx 6 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 30 -log './output/svmrank.pose.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.h5 -test_idx 6 -valid_idx 7 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 30 -log './output/svmrank.pose.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.h5 -test_idx 7 -valid_idx 0 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 30 -log './output/svmrank.pose.fs.txt'

#delta features
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.abs.h5 -test_idx 0 -valid_idx 1 -c 0.01,0.1,0.5,1.0 -log './output/svmrank.pose.delta.abs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.abs.h5 -test_idx 1 -valid_idx 2 -c 0.01,0.1,0.5,1.0 -log './output/svmrank.pose.delta.abs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.abs.h5 -test_idx 2 -valid_idx 3 -c 0.01,0.1,0.5,1.0 -log './output/svmrank.pose.delta.abs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.abs.h5 -test_idx 3 -valid_idx 4 -c 0.01,0.1,0.5,1.0 -log './output/svmrank.pose.delta.abs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.abs.h5 -test_idx 4 -valid_idx 5 -c 0.01,0.1,0.5,1.0 -log './output/svmrank.pose.delta.abs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.abs.h5 -test_idx 5 -valid_idx 6 -c 0.01,0.1,0.5,1.0 -log './output/svmrank.pose.delta.abs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.abs.h5 -test_idx 6 -valid_idx 7 -c 0.01,0.1,0.5,1.0 -log './output/svmrank.pose.delta.abs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.abs.h5 -test_idx 7 -valid_idx 0 -c 0.01,0.1,0.5,1.0 -log './output/svmrank.pose.delta.abs.txt'


#svm ranking exp using emo + pose features
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 0 -valid_idx 1 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 1 -valid_idx 2 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 2 -valid_idx 3 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 3 -valid_idx 4 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 4 -valid_idx 5 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 5 -valid_idx 6 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 6 -valid_idx 7 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 7 -valid_idx 0 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.txt'

#svm ranking exp using pose features but selected
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.h5 -test_idx 0 -valid_idx 1 -c 0.05,0.1,0.5,1.0,5.0,10.0 -log './output/svmrank.pose.fs.txt' --feat_select -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.h5 -test_idx 1 -valid_idx 2 -c 0.05,0.1,0.5,1.0,5.0,10.0 -log './output/svmrank.pose.fs.txt' --feat_select -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.h5 -test_idx 2 -valid_idx 3 -c 0.05,0.1,0.5,1.0,5.0,10.0 -log './output/svmrank.pose.fs.txt' --feat_select -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.h5 -test_idx 3 -valid_idx 4 -c 0.05,0.1,0.5,1.0,5.0,10.0 -log './output/svmrank.pose.fs.txt' --feat_select -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.h5 -test_idx 4 -valid_idx 5 -c 0.05,0.1,0.5,1.0,5.0,10.0 -log './output/svmrank.pose.fs.txt' --feat_select -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.h5 -test_idx 5 -valid_idx 6 -c 0.05,0.1,0.5,1.0,5.0,10.0 -log './output/svmrank.pose.fs.txt' --feat_select -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.h5 -test_idx 6 -valid_idx 7 -c 0.05,0.1,0.5,1.0,5.0,10.0 -log './output/svmrank.pose.fs.txt' --feat_select -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.h5 -test_idx 7 -valid_idx 0 -c 0.05,0.1,0.5,1.0,5.0,10.0 -log './output/svmrank.pose.fs.txt' --feat_select -max_feat 50

#svm ranking exp using emo + pose features but selected
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 0 -valid_idx 1 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.fs.txt' --feat_select -max_feat 50 
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 1 -valid_idx 2 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.fs.txt' --feat_select -max_feat 50 
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 2 -valid_idx 3 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.fs.txt' --feat_select -max_feat 50 
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 3 -valid_idx 4 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.fs.txt' --feat_select -max_feat 50 
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 4 -valid_idx 5 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.fs.txt' --feat_select -max_feat 50 
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 5 -valid_idx 6 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.fs.txt' --feat_select -max_feat 50 
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 6 -valid_idx 7 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.fs.txt' --feat_select -max_feat 50 
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 7 -valid_idx 0 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.fs.txt' --feat_select -max_feat 50 

#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 0 -valid_idx 1 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.fs.txt' --feat_select --pairwise -max_feat 50 
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 1 -valid_idx 2 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.fs.txt' --feat_select --pairwise -max_feat 50 
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 2 -valid_idx 3 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.fs.txt' --feat_select --pairwise -max_feat 50 
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 3 -valid_idx 4 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.fs.txt' --feat_select --pairwise -max_feat 50 
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 4 -valid_idx 5 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.fs.txt' --feat_select --pairwise -max_feat 50 
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 5 -valid_idx 6 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.fs.txt' --feat_select --pairwise -max_feat 50 
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 6 -valid_idx 7 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.fs.txt' --feat_select --pairwise -max_feat 50 
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.h5 -test_idx 7 -valid_idx 0 -c 0.1,0.5,1.0,5.0 -log './output/svmrank.emo.pose.fs.txt' --feat_select --pairwise -max_feat 50 

#svm ranking exp using emo + pose features but delta
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 0 -valid_idx 1 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 1 -valid_idx 2 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 2 -valid_idx 3 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 3 -valid_idx 4 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 4 -valid_idx 5 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 5 -valid_idx 6 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 6 -valid_idx 7 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 7 -valid_idx 0 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.txt'

#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 0 -valid_idx 1 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.fs.txt' --feat_select -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 1 -valid_idx 2 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.fs.txt' --feat_select -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 2 -valid_idx 3 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.fs.txt' --feat_select -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 3 -valid_idx 4 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.fs.txt' --feat_select -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 4 -valid_idx 5 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.fs.txt' --feat_select -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 5 -valid_idx 6 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.fs.txt' --feat_select -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 6 -valid_idx 7 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.fs.txt' --feat_select -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 7 -valid_idx 0 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.fs.txt' --feat_select -max_feat 50


#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 0 -valid_idx 1 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.fs.txt' --feat_select --pairwise -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 1 -valid_idx 2 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.fs.txt' --feat_select --pairwise -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 2 -valid_idx 3 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.fs.txt' --feat_select --pairwise -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 3 -valid_idx 4 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.fs.txt' --feat_select --pairwise -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 4 -valid_idx 5 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.fs.txt' --feat_select --pairwise -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 5 -valid_idx 6 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.fs.txt' --feat_select --pairwise -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 6 -valid_idx 7 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.fs.txt' --feat_select --pairwise -max_feat 50
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.POSE.func.indiv.nan.delta.h5 -test_idx 7 -valid_idx 0 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.pose.delta.fs.txt' --feat_select --pairwise -max_feat 50

#svm ranking exp using emo + pose features but delta
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.func.indiv.nan.delta.h5 -test_idx 0 -valid_idx 1 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.nv.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.func.indiv.nan.delta.h5 -test_idx 1 -valid_idx 2 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.nv.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.func.indiv.nan.delta.h5 -test_idx 2 -valid_idx 3 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.nv.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.func.indiv.nan.delta.h5 -test_idx 3 -valid_idx 4 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.nv.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.func.indiv.nan.delta.h5 -test_idx 4 -valid_idx 5 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.nv.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.func.indiv.nan.delta.h5 -test_idx 5 -valid_idx 6 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.nv.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.func.indiv.nan.delta.h5 -test_idx 6 -valid_idx 7 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.nv.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.func.indiv.nan.delta.h5 -test_idx 7 -valid_idx 0 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.nv.delta.txt'

#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.func.indiv.nan.delta.h5 -test_idx 0 -valid_idx 1 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 50 -log './output/svmrank.emo.nv.delta.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.func.indiv.nan.delta.h5 -test_idx 1 -valid_idx 2 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 50 -log './output/svmrank.emo.nv.delta.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.func.indiv.nan.delta.h5 -test_idx 2 -valid_idx 3 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 50 -log './output/svmrank.emo.nv.delta.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.func.indiv.nan.delta.h5 -test_idx 3 -valid_idx 4 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 50 -log './output/svmrank.emo.nv.delta.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.func.indiv.nan.delta.h5 -test_idx 4 -valid_idx 5 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 50 -log './output/svmrank.emo.nv.delta.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.func.indiv.nan.delta.h5 -test_idx 5 -valid_idx 6 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 50 -log './output/svmrank.emo.nv.delta.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.func.indiv.nan.delta.h5 -test_idx 6 -valid_idx 7 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 50 -log './output/svmrank.emo.nv.delta.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.func.indiv.nan.delta.h5 -test_idx 7 -valid_idx 0 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 50 -log './output/svmrank.emo.nv.delta.fs.txt'
#emotio feature analysis
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.func.indiv.nan.min.h5 --feature_analysis --pairwise -log './output/svmrank.emo.analysis.txt'

#svm ranking exp using emo + nv + pose features but delta
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.POSE.func.indiv.nan.delta.h5 -test_idx 0 -valid_idx 1 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.nv.pose.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.POSE.func.indiv.nan.delta.h5 -test_idx 1 -valid_idx 2 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.nv.pose.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.POSE.func.indiv.nan.delta.h5 -test_idx 2 -valid_idx 3 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.nv.pose.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.POSE.func.indiv.nan.delta.h5 -test_idx 3 -valid_idx 4 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.nv.pose.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.POSE.func.indiv.nan.delta.h5 -test_idx 4 -valid_idx 5 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.nv.pose.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.POSE.func.indiv.nan.delta.h5 -test_idx 5 -valid_idx 6 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.nv.pose.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.POSE.func.indiv.nan.delta.h5 -test_idx 6 -valid_idx 7 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.nv.pose.delta.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.POSE.func.indiv.nan.delta.h5 -test_idx 7 -valid_idx 0 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.emo.nv.pose.delta.txt'

#selected
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.POSE.func.indiv.nan.delta.h5 -test_idx 0 -valid_idx 1 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 50 -log './output/svmrank.emo.nv.pose.delta.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.POSE.func.indiv.nan.delta.h5 -test_idx 1 -valid_idx 2 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 50 -log './output/svmrank.emo.nv.pose.delta.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.POSE.func.indiv.nan.delta.h5 -test_idx 2 -valid_idx 3 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 50 -log './output/svmrank.emo.nv.pose.delta.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.POSE.func.indiv.nan.delta.h5 -test_idx 3 -valid_idx 4 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 50 -log './output/svmrank.emo.nv.pose.delta.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.POSE.func.indiv.nan.delta.h5 -test_idx 4 -valid_idx 5 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 50 -log './output/svmrank.emo.nv.pose.delta.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.POSE.func.indiv.nan.delta.h5 -test_idx 5 -valid_idx 6 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 50 -log './output/svmrank.emo.nv.pose.delta.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.POSE.func.indiv.nan.delta.h5 -test_idx 6 -valid_idx 7 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 50 -log './output/svmrank.emo.nv.pose.delta.fs.txt'
#python ./ranking_svm_exp.py -dt ../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.EMO.NV.POSE.func.indiv.nan.delta.h5 -test_idx 7 -valid_idx 0 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 50 -log './output/svmrank.emo.nv.pose.delta.fs.txt'

