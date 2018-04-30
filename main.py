import csv
import Feature_extraction as urlfeature
import trainer as tr
import warnings
warnings.filterwarnings("ignore")


def resultwriter(feature, output_dest):
    flag = True
    with open(output_dest, 'w') as f:
        for item in feature:
            w = csv.DictWriter(f, item[1].keys())
            if flag:
                w.writeheader()
                flag = False
            w.writerow(item[1])


def process_URL_list(file_dest,
                     output_dest):
    feature = []
    with open(file_dest) as file:
        for line in file:
            url = line.split(',')[0].strip()
            malicious_bool = line.split(',')[1].strip()
            if url != '':
                print('working on: ' + url)  # showoff
                ret_dict = urlfeature.feature_extract(url)
                ret_dict['malicious'] = malicious_bool
                feature.append([url, ret_dict]);
    resultwriter(feature, output_dest)


def process_test_list(file_dest,
                      output_dest):  # i think this takes whole file of urls without given malicious to extract their  feature and doest not provide malicious column like this will take query.txt
    global f
    feature = []
    with open(file_dest) as file:
        for line in file:
            url = line.strip()
            if url != '':
                print('working on: ' + url)  # showoff
                ret_dict = urlfeature.feature_extract(url)
                feature.append([url, ret_dict]);

    resultwriter(feature, output_dest)


# change
def process_test_url(url,
                     output_dest):
    feature = []
    url = url.strip()
    if url != '':
        print('working on: ' + url)
        ret_dict = urlfeature.feature_extract(url)
        feature.append([url, ret_dict])
    resultwriter(feature, output_dest)


def main():
     for i in range(1, 6):
         s = 'comp/train_Data' + str(i) + '.csv'
         k = 'comp/test_features' + str(i) + '.csv'
         tr.train(s, k)
         print(' --------------------------------------------------------  ')
