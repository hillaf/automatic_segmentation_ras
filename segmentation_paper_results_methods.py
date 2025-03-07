import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import scale, normalize
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

from collections import defaultdict

from pycocotools import mask as mask_utils
from pycocotools import _mask as coco_mask



def features_from_masks(imgfile, img_path, df, category_name, img_files):  
    mask_img_file = img_path + imgfile + '.png'
    image = cv2.imread(mask_img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if len(img_files[img_files['file_name'] == imgfile + '.png']['id']) > 0:
        image_id = img_files[img_files['file_name'] == imgfile + '.png']['id'].values[0]
    else:
        print(imgfile, 'masks not found for class', category_name)
        return ([],[],[],[],[],[])

    data = df[(df['image_id'] == image_id) & (df['category_name'] == category_name)]
    h, w, channels = image.shape

    masks_data = data
    minors = []
    majors = []
    areas = []
    n_contours = []
    n_conv_defects = []
    moments = []
    
    for i, seg in masks_data.iterrows():
        rle_dict = {"counts": seg["segmentation"], "size": [h, w]}
        # decode wants strings
        compressed_rle = mask_utils.frPyObjects(rle_dict, rle_dict.get('size')[0], rle_dict.get('size')[1])
        m = mask_utils.decode(compressed_rle)

        # a little less conversation
        d = m.astype("uint8") * 255      
        rgb_img = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
        img_uint8 = cv2.convertScaleAbs(rgb_img)
        imgray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        n_contours.append(len(contours))
        hull = cv2.convexHull(np.array([point for cnt in contours for point in cnt]))
                
        if len(hull) > 5:
            (x,y),(major_a,minor_a),angle = cv2.fitEllipse(hull)
            # nb: filter one bad ellipse from the dataset
            minors.append(minor_a)
            majors.append(major_a)
        else:
            print("too few points in hull to fit ellipse", mask_img_file)
            minors.append(-1)
            majors.append(-1)

        area = sum([cv2.contourArea(c) for c in contours])
        areas.append(area)
        
        # convexity defects
        n_defects = 0
        for cnt in contours:
            hull = cv2.convexHull(np.array([point for point in cnt]), returnPoints=False)
            hull[::-1].sort(axis=0)
            defects = cv2.convexityDefects(cnt, hull)
            if defects is not None:
                n_defects = n_defects + len(defects)
        n_conv_defects.append(n_defects)

        m = cv2.moments(contours[0])
        humoments = cv2.HuMoments(m).flatten()
        moments.append(humoments)
    return majors, minors, areas, n_contours, n_conv_defects, moments


def compute_mask_metrics(imgs, imgs_path, df, img_files):
    cats = ['fish', 'bad', 'head', 'double']
    all_majors = defaultdict(list)
    all_minors = defaultdict(list)
    all_areas = defaultdict(list)
    all_n_contours = defaultdict(list)
    all_conv_defects = defaultdict(list)
    all_moments = defaultdict(list)
    for i, imgfile in enumerate(imgs):
        image = cv2.imread(imgs_path + imgfile + '.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for cat in cats:
            majors, minors, areas, n_contours, conv_defects, moments = features_from_masks(imgfile, imgs_path, df, cat, img_files)
            all_majors[cat].extend(majors)
            all_minors[cat].extend(minors)
            all_areas[cat].extend(areas)
            all_n_contours[cat].extend(n_contours)
            all_conv_defects[cat].extend(conv_defects)
            all_moments[cat].extend(moments)
    return [all_majors, all_minors, all_areas, all_n_contours, all_conv_defects, all_moments]


def mvals(moms, i):
    return moms[[i]].values.flatten()


def get_features(metrics_list, classes):
    maj, mino, areas, c_cont, c_conv, momss = metrics_list
    alldf = []
    for cl in classes:
        moms = pd.DataFrame(momss[cl])
        df = pd.DataFrame([maj[cl], mino[cl], areas[cl], c_cont[cl], c_conv[cl], mvals(moms, 0), mvals(moms, 1), \
                           mvals(moms, 2), mvals(moms, 3), mvals(moms, 4), mvals(moms, 5), mvals(moms, 6)]).T
        alldf.append(df)
    return alldf


def get_feature_array(dfs):
    y = list(np.repeat(0, len(dfs[0]))) + list(np.repeat(1, len(dfs[1]))) + list(np.repeat(2, len(dfs[2]))) + list(np.repeat(3, len(dfs[3])))
    X = np.array(scale(pd.concat(dfs).to_numpy()))
    return X, y


def cross_validate(X, y, class_i, rs, classes):
    classifiers = [
        KNeighborsClassifier(3),
        #SVC(gamma=3, C=1, random_state=rs),
        #SVC(gamma=2, C=1, random_state=rs),
        #GaussianProcessClassifier(1.0 * RBF(1.0), random_state=rs),
        #DecisionTreeClassifier(max_depth=20, random_state=rs),
        #DecisionTreeClassifier(max_depth=50, random_state=rs),
        MLPClassifier(alpha=0.05, max_iter=5000, random_state=rs),
        #MLPClassifier(alpha=0.01, max_iter=5000, random_state=rs),
        #AdaBoostClassifier(random_state=rs),
        #GaussianNB(),
        #QuadraticDiscriminantAnalysis(),
        #RandomForestClassifier(max_depth=10, n_estimators=20, max_features=2, random_state=rs),
        RandomForestClassifier(max_depth=10, n_estimators=20, max_features=3, random_state=rs),
        #RandomForestClassifier(max_depth=10, n_estimators=20, max_features=3, random_state=rs),
    ]
    print('One-vs-all binary classification, class', class_i, ':', classes[class_i], 'vs all')
    # commented: all classes. not commented and classes specified: binary one-vs-all.
    y = [0 if cl == class_i else 1 for cl in y]
    for clf in classifiers:
        print('Cross-validation, model', clf)
        scores = cross_val_score(clf, X, y, cv=5)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


def evaluate_improvement(predict_result, y_test, mode):
    correct_fish = 0
    correct_bad = 0
    false_positives = 0
    false_negatives = 0
    for i, p in enumerate(predict_result):
        true_class = y_test[i]
        if mode == 'fish':
            # correct prediction
            if true_class == p:
                if p == 0:
                    correct_fish = correct_fish + 1
                else:
                    correct_bad = correct_bad + 1
            # false negative
            elif true_class == 0:
                false_negatives = false_negatives + 1
            elif true_class != 0:
                false_positives = false_positives + 1
            else:
                print('shouldnt happen')
        elif mode == 'fish+head':
            if true_class == p:
                if p == 0:
                    correct_fish = correct_fish + 1
                elif p == 2:
                    correct_fish = correct_fish + 1
                else:
                    correct_bad = correct_bad + 1
            # we don't mind mixing up heads and whole fish for this metric
            elif (true_class == 0 or true_class == 2) and (p == 0 or p == 2):
                correct_fish = correct_fish + 1 
            # other cases either false positive or false negative
            else:
                if (true_class == 0 or true_class == 2):
                    false_negatives = false_negatives + 1
                elif (true_class != 0 and true_class != 2):
                    false_positives = false_positives + 1
                else:
                    print('shouldnt happen')
        elif mode == 'fish+head+multiple':
            if true_class == p:
                if p == 1:
                    correct_bad = correct_bad + 1
                else:
                    correct_fish = correct_fish + 1
            else:
                if true_class == 1:
                    false_positives = false_positives + 1
                elif true_class != 1:
                    false_negatives = false_negatives + 1
                else:
                    print('shouldnt happen')
    filtering_result = (correct_fish) / (correct_fish + false_positives)
    return filtering_result


def cross_validate_filtering_results(X, y, rs, prompt_method):
    X = np.array(X)
    y = np.array(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    clf = RandomForestClassifier(max_depth=10, n_estimators=20, max_features=3, random_state=rs)
    scores_f = []
    scores_fh = []
    scores_fhm = []
    conf_matrix_list_of_arrays = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        clf.fit(X[train_index], y[train_index])
        predictions = clf.predict(X[test_index])
        score = evaluate_improvement(predictions, y[test_index], mode='fish')
        scores_f.append(score)
        score = evaluate_improvement(predictions, y[test_index], mode='fish+head')
        scores_fh.append(score)
        score = evaluate_improvement(predictions, y[test_index], mode='fish+head+multiple')
        scores_fhm.append(score)

        cm = confusion_matrix(y[test_index], predictions, labels=clf.classes_)
        conf_matrix_list_of_arrays.append(cm)
    mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=mean_of_conf_matrix_arrays,
                                  display_labels=['fish', 'bad', 'head', 'multiple'])
    disp.plot(ax=ax, values_format='.0f')
    plt.title(prompt_method)
    plt.show()
    return scores_f, scores_fh, scores_fhm


def cross_validate_baseline_filtering_results(df, rs, filter_max, filter_min):
    areas = df['area'].values
    true_classes = [i-1 for i in df['category_id'].values]
    print(true_classes.count(0))
    print(true_classes.count(1))
    print(true_classes.count(2))
    print(true_classes.count(3))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    scores_f = []
    scores_fh = []
    scores_fhm = []
    print('N', len(true_classes))
    for i, (train_index, test_index) in enumerate(skf.split(areas, true_classes)):
        print('Train/test split', len(train_index), '/', len(test_index))
        filtered_masks = []
        for j in test_index:
            if (areas[j] < filter_max) and (areas[j] > filter_min):
                filtered_masks.append(true_classes[j])
        scores_f.append((len([i for i in filtered_masks if i == 0]) / len(filtered_masks)))
        scores_fh.append((len([i for i in filtered_masks if (i == 0 or i == 2)]) / len(filtered_masks)))
        scores_fhm.append((len([i for i in filtered_masks if i != 1]) / len(filtered_masks)))
    return scores_f, scores_fh, scores_fhm
