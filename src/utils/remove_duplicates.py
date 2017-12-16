import os

os.chdir('..')
os.chdir('..')

train2 = os.path.join(os.getcwd(), 'dataset', 'train2')
test = os.path.join(os.getcwd(), 'dataset', 'test')
test_ground_truth = os.path.join(os.getcwd(), 'dataset', 'test_ground_truth')

for file in os.listdir(train2):
    if os.path.exists(os.path.join(test, file)):
        os.remove(os.path.join(test, file))

    if os.path.exists(os.path.join(test_ground_truth, file)):
        os.remove(os.path.join(test_ground_truth, file))
