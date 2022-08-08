import numpy as np
import pandas as pd
import argparse

def main_gpu(points, n_samples):

    # Take note of the number of data (rows)
    n_points = len(points)

    # The array of indices [0, 1, 2, ...]
    idx = cp.arange(n_points)

    # A boolean array for which data is picked or not
    # True = not picked, False = picked
    available_to_pick = cp.ones(n_points, dtype=bool)

    # An array to note the minimum distances to all data
    # Initially set to infinity for all data
    dists = cp.full(n_points, cp.inf)

    # Add the first data (index 0) to the picked list
    latest_pick = 0
    available_to_pick[latest_pick] = False

    # This array is used to store the index of the picked data in order
    # The first one is already index 0
    ordered_picks = cp.zeros(n_samples, dtype=int)

    # Move the data to GPU
    points = cp.asarray(points)

    # Loop to get n_samples, one sample per iteration
    for i in range(1, n_samples):

        # Calculate distances from the latest picked data to
        # every other unpicked data (available-to-pick data)
        dist_to_last_added_point = cp.linalg.norm(
            points[available_to_pick] - points[latest_pick], axis=1
        )

        # Update the list of minimum distances
        dists[available_to_pick] = cp.minimum(
            dist_to_last_added_point, dists[available_to_pick]
        )

        # Find the index of the data whose distances is
        # the farthest from every picked data
        # Take this data as the latest pick
        selected = cp.argmax(dists[available_to_pick])
        latest_pick = idx[available_to_pick][selected]

        # Mark this data as picked and add its index to the order list
        available_to_pick[latest_pick] = False
        ordered_picks[i] = latest_pick

    # Return 1: The inverted available-to-pick list
    #           i.e. True = picked, False = not picked
    # Return 2: The list of indices of the picked data in order
    return ~available_to_pick.get(), ordered_picks.get()

def main_cpu(points, n_samples):

    # Take note of the number of data (rows)
    n_points = len(points)

    # The array of indices [0, 1, 2, ...]
    idx = np.arange(n_points)

    # A boolean array for which data is picked or not
    # True = not picked, False = picked
    available_to_pick = np.ones(n_points, dtype=bool)

    # An array to note the minimum distances to all data
    # Initially set to infinity for all data
    dists = np.full(n_points, np.inf)

    # Add the first data (index 0) to the picked list
    latest_pick = 0
    available_to_pick[latest_pick] = False

    # This array is used to store the index of the picked data in order
    # The first one is already index 0
    ordered_picks = np.zeros(n_samples, dtype=int)

    # Move the data to GPU
    points = np.asarray(points)

    # Loop to get n_samples, one sample per iteration
    for i in range(1, n_samples):

        # Calculate distances from the latest picked data to
        # every other unpicked data (available-to-pick data)
        dist_to_last_added_point = np.linalg.norm(
            points[available_to_pick] - points[latest_pick], axis=1
        )

        # Update the list of minimum distances
        dists[available_to_pick] = np.minimum(
            dist_to_last_added_point, dists[available_to_pick]
        )

        # Find the index of the data whose distances is
        # the farthest from every picked data
        # Take this data as the latest pick
        selected = np.argmax(dists[available_to_pick])
        latest_pick = idx[available_to_pick][selected]

        # Mark this data as picked and add its index to the order list
        available_to_pick[latest_pick] = False
        ordered_picks[i] = latest_pick

    # Return 1: The inverted available-to-pick list
    #           i.e. True = picked, False = not picked
    # Return 2: The list of indices of the picked data in order
    return ~available_to_pick, ordered_picks


def normalize_data (data,mean,std):
    return (data - mean)/std

def rescale_data(data,mean,std):
    return data*std+mean

if __name__ == "__main__":

    fulldescription = """
        Use farthest point sampling to generate an ordering of data from a csv file,
        from "most unique" to "least unique" datapoints. This code assumes there is a single
        column to label the data, and the rest of the columns are descriptors, which will be
        used to perform the sampling. This code can run either using the gpu or cpu,
        implemented using either cupy or numpy.
    """

    parser = argparse.ArgumentParser(description=fulldescription)
    parser.add_argument('csv_file', type=str,
                        help='CSV file containing the descriptors to use for sampling')
    parser.add_argument('n_samples', type=int,
                        help='The number of samples for the farthest point sampling')
    parser.add_argument('label_column', type=str,
                        help='The name of the csv column with the labels for the data (e.g., filename)')
    parser.add_argument("-gpu", action="store_true",
                        help="If this flag is present, " +
                        " the code will run on the GPU using cupy")

    args = parser.parse_args()
    csv_file = args.csv_file
    n_samples = args.n_samples
    label_column = args.label_column
    use_gpu = args.gpu

    data = pd.read_csv(csv_file)
	
    ## choosing the diverse set from the train data, one can change the weights
    # Also, remove non-descriptor columns and columns with a std of 0
    unnamed_cols = [col for col in data.columns if 'Unnamed' in col]
    x_data = data.drop([label_column] + unnamed_cols, axis=1)
    x_data = [x for x in x_data if np.std(data[x]) != 0]
    points = data[x_data].values
    x_mu = np.mean(points, axis=0)
    x_std = np.std(points, axis=0)
    points = normalize_data(points,x_mu,x_std)
   
    print(f"Number of descriptors being used for sampling: {len(x_data)}")
    print("List of descriptors being used for sampling:")
    for x in x_data:
        print(x)
 
    if use_gpu:
        import cupy as cp
        results_bool, results_order = main_gpu(points, n_samples)
    else:
        results_bool, results_order = main_cpu(points, n_samples)
    
    # Write the results to numpy arrays
    np.save("bool_pick.npy", results_bool)
    np.save("order_pick.npy", results_order)
    
    # Write the results to a new csv file (order dictatates the order in
    # which each structure was sampled. Bool indicates whether a structure was
    # sampled in the given number of samples.
    data['order_pick'] = n_samples + 1
    for count, idx in enumerate(results_order):
        data.iloc[idx, data.columns.get_loc('order_pick')] = count

    data['bool_pick'] = results_bool

    data.to_csv("{}_with_picks.csv".format(csv_file.split('.csv')[0]))

