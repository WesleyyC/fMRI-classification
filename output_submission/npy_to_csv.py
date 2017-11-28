import sys
import numpy as np

if len(sys.argv) < 3:
    print(
        "Usage: python3 subset_acc.py [subset|auc] <prediction_file> <output_file>", file=sys.stderr)
    print("e.g.\tpython3 npy_to_csv.py subset result.npy subset_submission.csv", file=sys.stderr)
    print("\tpython3 npy_to_csv.py auc result.npy auc_submission.csv", file=sys.stderr)
    sys.exit(1)

prediction = np.load(sys.argv[2]).astype(int)
if(prediction.shape != (1971, 19)):
    print("Labels file is not in the correct dimension", file=sys.stderr)
    sys.exit(1)

tag_name = ['faces', 'left_foot', 'visual_digits', 'left_hand', 'calculation',
            'language', 'horizontal_checkerboard', 'human_sound',
            'vertical_checkerboard', 'objects', 'places', 'scramble',
            'right_hand', 'right_foot', 'visual_words', 'visual',
            'non_human_sound', 'auditory', 'saccades']


if sys.argv[1].lower() == "subset":
    # begin writing to subset accuracy output file
    with open(sys.argv[3], 'w') as f:
        print("id,tags", file=f)
        for i in range(1971):
            label = []
            for j in range(19):
                if prediction[i, j]:
                    label.append(tag_name[j])
            print("%d,%s" % (i, ' '.join(label)), file=f)
else:
    # write to auc output file
    # add the id column
    prediction = np.concatenate(
        (np.arange(prediction.shape[0]).reshape((prediction.shape[0], 1)), prediction), axis=1)
    np.savetxt(sys.argv[3], prediction, header="id,%s" %
               ",".join(tag_name), delimiter=",", fmt='%d', comments='')
