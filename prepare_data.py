from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from argparse import ArgumentParser
import numpy as np
from sklearn.decomposition import PCA
import os
import pandas as pd
import time
from requests.exceptions import (
    ChunkedEncodingError,
    RequestException,
    ConnectionError,
    Timeout,
)


def fetch_data_with_retry(data_dir, pipeline, quality_checked, max_retries=5, delay=5):
    for i in range(max_retries):
        try:
            abide = datasets.fetch_abide_pcp(
                data_dir=data_dir,
                pipeline=pipeline,
                quality_checked=quality_checked,
                legacy_format=False,
            )
            return abide
        except (ChunkedEncodingError, RequestException, ConnectionError) as e:
            print(f"Network error occurred: {e}. Retry {i+1} of {max_retries}")
            time.sleep(delay)
        except Timeout as e:
            print(f"Timeout error occurred: {e}. Retry {i+1} of {max_retries}")
            time.sleep(delay)
        except Exception as e:
            print(f"Unknown error occurred: {e}. Retry {i+1} of {max_retries}")
            time.sleep(delay)
    raise Exception("Failed to fetch data after multiple retries")


def prepare_data(data_dir, output_dir, pipeline="cpac", quality_checked=True):
    # get dataset

    print(f"pipeline {pipeline}")
    print("Loading dataset...")
    abide = fetch_data_with_retry(data_dir, pipeline, quality_checked)

    print(f"Downloaded all files")

    # make list of filenames
    fmri_filenames = abide.func_preproc

    # load atlas
    multiscale = datasets.fetch_atlas_basc_multiscale_2015()
    atlas_filename = multiscale.scale064

    # initialize masker object
    masker = NiftiLabelsMasker(
        labels_img=atlas_filename,
        standardize="zscore_sample",
        memory="nilearn_cache",
        verbose=0,
    )

    # initialize correlation measure
    correlation_measure = ConnectivityMeasure(
        kind="correlation",
        vectorize=True,
        discard_diagonal=True,
        standardize="zscore_sample",
    )

    try:  # check if feature file already exists
        # load features
        feat_file = os.path.join(output_dir, "ABIDE_BASC064_features.npz")
        X_features = np.load(feat_file)["a"]
        print("Feature file found.")

    except:  # if not, extract features
        X_features = []  # To contain upper half of matrix as 1d array
        print("No feature file found. Extracting features...")

        for i, sub in enumerate(fmri_filenames):
            # extract the timeseries from the ROIs in the atlas
            time_series = masker.fit_transform(sub)
            print(time_series)
            print(time_series.shape)
            print("------------------------------")
            # create a region x region correlation matrix
            correlation_matrix = correlation_measure.fit_transform([time_series])[0]
            print(correlation_matrix)
            print(correlation_matrix.shape)
            print("------------------------------")
            # add to our container
            X_features.append(correlation_matrix)
            # keep track of status
            print("finished extracting %s of %s" % (i + 1, len(fmri_filenames)))
        # Save features
        np.savez_compressed(
            os.path.join(output_dir, "ABIDE_BASC064_features"), a=X_features
        )

    print(X_features[0].shape)
    print(len(X_features))
    print(X_features[0])
    print(X_features)
    # Dimensionality reduction of features with PCA
    print("Running PCA...")
    pca = PCA(0.99).fit(X_features)  # keeping 99% of variance
    X_features_pca = pca.transform(X_features)

    # Transform phenotypic data into dataframe
    abide_pheno = pd.DataFrame(abide.phenotypic)

    # Get the target vector
    y_target = abide_pheno["DX_GROUP"]

    return (X_features_pca, y_target)


def run():
    description = "Prepare data for classifier on the ABIDE data to predict autism"
    parser = ArgumentParser(__file__, description)
    parser.add_argument(
        "data_dir",
        action="store",
        help="""Path to the data directory that contains the
                        ABIDE data set. If you already have the data set, this
                        should be the folder that contains the subfolder
                        'ABIDE_pcp'. If this folder does not exists yet, it will
                        be created in the directory you provide.""",
    )
    parser.add_argument(
        "output_dir",
        action="store",
        help="""Path to the directory where you want to store
                        outputs.""",
    )
    parser.add_argument(
        "pipeline", action="store", help="Pipline used for preprocessing"
    )
    args = parser.parse_args()

    X_features_pca, y_target = prepare_data(
        args.data_dir, args.output_dir, args.pipeline
    )

    return (X_features_pca, y_target)


if __name__ == "__main__":
    run()
