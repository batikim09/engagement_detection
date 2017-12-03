#sanity checks to see if the code works in your virtual environment, be careful about the location of feature DB(.h5)
#svm ranking exp using pose features without feature selection
python ./ranking_svm_exp.py -dt ./feature/DEEPCP.POSE.func.indiv.nan.delta.h5 -test_idx 0 -valid_idx 1 -c 0.01,0.1,1.0,10.0 -log './output/svmrank.pose.txt'

echo 'the output should be like:
Complexity 0.010, performance of ranking on training data: tau_d: 0.354, prob_cm: [[ 0.64822134  0.35177866]
 [ 0.35714286  0.64285714]]
Complexity 0.010, performance of ranking on validataion data: tau_d: 0.524, prob_cm: [[ 0.47169811  0.52830189]
 [ 0.52        0.48      ]]
Complexity 0.100, performance of ranking on training data: tau_d: 0.330, prob_cm: [[ 0.67786561  0.32213439]
 [ 0.33730159  0.66269841]]
Complexity 0.100, performance of ranking on validataion data: tau_d: 0.515, prob_cm: [[ 0.50943396  0.49056604]
 [ 0.54        0.46      ]]
Complexity 1.000, performance of ranking on training data: tau_d: 0.366, prob_cm: [[ 0.61264822  0.38735178]
 [ 0.3452381   0.6547619 ]]
Complexity 1.000, performance of ranking on validataion data: tau_d: 0.500, prob_cm: [[ 0.46226415  0.53773585]
 [ 0.46        0.54      ]]
Complexity 10.000, performance of ranking on training data: tau_d: 0.393, prob_cm: [[ 0.60474308  0.39525692]
 [ 0.39087302  0.60912698]]
Complexity 10.000, performance of ranking on validataion data: tau_d: 0.471, prob_cm: [[ 0.51886792  0.48113208]
 [ 0.46        0.54      ]]
validation: best complexity: 10.000 and its performance: 0.471, prob_cm: [[ 0.51886792  0.48113208]
 [ 0.46        0.54      ]]
performance of ranking on testing data: tau_d: 0.533, prob_cm: [[ 0.40322581  0.59677419]
 [ 0.46666667  0.53333333]]
'

#pairwise loss based feature ranking and selection
python ./ranking_svm_exp.py -dt ./feature/DEEPCP.POSE.func.indiv.nan.delta.h5 -test_idx 0 -valid_idx 1 -c 0.01,0.1,1.0,10.0 --feat_select --pairwise -max_feat 30 -log './output/svmrank.pose.fs.txt'

echo 'the output should be like:
Complexity 0.010, performance of ranking on training data: tau_d: 0.373, prob_cm: [[ 0.6284585  0.3715415]
 [ 0.375      0.625    ]]
Complexity 0.010, performance of ranking on validataion data: tau_d: 0.388, prob_cm: [[ 0.64150943  0.35849057]
 [ 0.42        0.58      ]]
Complexity 0.100, performance of ranking on training data: tau_d: 0.436, prob_cm: [[ 0.56521739  0.43478261]
 [ 0.43650794  0.56349206]]
Complexity 0.100, performance of ranking on validataion data: tau_d: 0.476, prob_cm: [[ 0.54716981  0.45283019]
 [ 0.5         0.5       ]]
Complexity 1.000, performance of ranking on training data: tau_d: 0.447, prob_cm: [[ 0.53162055  0.46837945]
 [ 0.42460317  0.57539683]]
Complexity 1.000, performance of ranking on validataion data: tau_d: 0.456, prob_cm: [[ 0.52830189  0.47169811]
 [ 0.44        0.56      ]]
Complexity 10.000, performance of ranking on training data: tau_d: 0.529, prob_cm: [[ 0.47628458  0.52371542]
 [ 0.53373016  0.46626984]]
Complexity 10.000, performance of ranking on validataion data: tau_d: 0.534, prob_cm: [[ 0.38679245  0.61320755]
 [ 0.45        0.55      ]]
validation: best complexity: 0.010 and its performance: 0.388, prob_cm: [[ 0.64150943  0.35849057]
 [ 0.42        0.58      ]]
performance of ranking on testing data: tau_d: 0.607, prob_cm: [[ 0.35483871  0.64516129]
 [ 0.56666667  0.43333333]]
'


#Random forest based feature ranking and selection but using validataion data for its selection.
python ./ranking_svm_exp.py -dt ./feature/DEEPCP.POSE.func.indiv.nan.delta.h5 -test_idx 0 -valid_idx 1 -c 0.01,0.1,1.0,10.0 --feat_select -max_feat 30 -rf_n 500 --fs_by_valid --fs_by_rf -log './output/svmrank.pose.delta.fs_by_rf.30.txt'

echo 'the output should be like:
Complexity 0.010, performance of ranking on training data: tau_d: 0.460, prob_cm: [[ 0.51383399  0.48616601]
 [ 0.43452381  0.56547619]]
Complexity 0.010, performance of ranking on validataion data: tau_d: 0.447, prob_cm: [[ 0.55660377  0.44339623]
 [ 0.45        0.55      ]]
Complexity 0.100, performance of ranking on training data: tau_d: 0.474, prob_cm: [[ 0.50197628  0.49802372]
 [ 0.45039683  0.54960317]]
Complexity 0.100, performance of ranking on validataion data: tau_d: 0.398, prob_cm: [[ 0.5754717  0.4245283]
 [ 0.37       0.63     ]]
Complexity 1.000, performance of ranking on training data: tau_d: 0.466, prob_cm: [[ 0.51581028  0.48418972]
 [ 0.4484127   0.5515873 ]]
Complexity 1.000, performance of ranking on validataion data: tau_d: 0.471, prob_cm: [[ 0.54716981  0.45283019]
 [ 0.49        0.51      ]]
Complexity 10.000, performance of ranking on training data: tau_d: 0.470, prob_cm: [[ 0.52766798  0.47233202]
 [ 0.46825397  0.53174603]]
Complexity 10.000, performance of ranking on validataion data: tau_d: 0.451, prob_cm: [[ 0.56603774  0.43396226]
 [ 0.47        0.53      ]]
validation: best complexity: 0.100 and its performance: 0.398, prob_cm: [[ 0.5754717  0.4245283]
 [ 0.37       0.63     ]]
performance of ranking on testing data: tau_d: 0.582, prob_cm: [[ 0.40322581  0.59677419]
 [ 0.56666667  0.43333333]]
'
