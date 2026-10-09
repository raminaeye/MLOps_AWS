"""Preprocess Amazon review data for the BlazingText algorithm.

Used as the processing script for the SageMaker SKLearnProcessor job in
Lesson 2, Exercise 4. Reads a zipped JSON-lines file of Amazon reviews,
labels each review as helpful/unhelpful from its vote counts, splits reviews
into individual sentences (keeping the label), and writes 90/10 train/test
files in the BlazingText supervised format to the SageMaker processing
output directories.
"""
import json
import zipfile


def unzip_data(input_data_path):
    """Extract a zip archive into the current directory.

    Returns the name of the first file in the archive.
    """
    with zipfile.ZipFile(input_data_path, 'r') as input_data_zip:
        input_data_zip.extractall('.')
        return input_data_zip.namelist()[0]


# Input data is a file with a single JSON object per line with the following format:
# {
#  "reviewerID": <string>,
#  "asin": <string>,
#  "reviewerName": <string>,
#  "helpful": [
#    <int>, (indicating number of "helpful votes")
#    <int>  (indicating total number of votes)
#  ],
#  "reviewText": "<string>",
#  "overall": <int>,
#  "summary": "<string>",
#  "unixReviewTime": <int>,
#  "reviewTime": "<string>"
# }
#
# We are specifically interested in the fields "helpful" and "reviewText"
#

def label_data(input_data):
    """Label reviews based on the ratio of helpful votes.

    A review is labeled "__label__1" (helpful) if more than half of its
    votes are helpful, "__label__2" (unhelpful) if fewer than half are,
    and dropped if the ratio is exactly one half or there are no votes.
    Returns a list of "label reviewText" strings.
    """
    labeled_data = []
    HELPFUL_LABEL = "__label__1"
    UNHELPFUL_LABEL = "__label__2"

    for l in open(input_data, 'r'):
        l_object = json.loads(l)
        helpful_votes = float(l_object['helpful'][0])
        total_votes = l_object['helpful'][1]
        reviewText = l_object['reviewText']
        if total_votes != 0:
            if helpful_votes / total_votes > .5:
                labeled_data.append(" ".join([HELPFUL_LABEL, reviewText]))
            elif helpful_votes / total_votes < .5:
                labeled_data.append(" ".join([UNHELPFUL_LABEL, reviewText]))

    return labeled_data


# Labeled data is a list of sentences, starting with the label defined in label_data.

def split_sentences(labeled_data):
    """Split each labeled review into sentences, keeping the review's label.

    Splits on "." and drops empty fragments (common with "..."). Returns a
    list of "label sentence" strings.
    """
    new_split_sentences = []
    for d in labeled_data:
        label = d.split()[0]
        sentences = " ".join(d.split()[1:]).split(".")  # Initially split to separate label, then separate sentences
        for s in sentences:
            if s:  # Make sure sentences isn't empty. Common w/ "..."
                new_split_sentences.append(" ".join([label, s]))
    return new_split_sentences


def write_data(data, train_path, test_path, proportion):
    """Write the first `proportion` of lines to the train file, the rest to test."""
    border_index = int(proportion * len(data))
    with open(train_path, 'w') as train_f, open(test_path, 'w') as test_f:
        index = 0
        for d in data:
            if index < border_index:
                train_f.write(d + '\n')
            else:
                test_f.write(d + '\n')
            index += 1


if __name__ == "__main__":
    unzipped_path = unzip_data('/opt/ml/processing/input/reviews_Musical_Instruments_5.json.zip')
    labeled_data = label_data(unzipped_path)
    new_split_sentence_data = split_sentences(labeled_data)
    write_data(new_split_sentence_data, '/opt/ml/processing/output/train/hello_blaze_train_scikit', '/opt/ml/processing/output/test/hello_blaze_test_scikit', .9)
