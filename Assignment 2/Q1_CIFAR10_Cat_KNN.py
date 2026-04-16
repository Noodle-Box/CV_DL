import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
from skimage import color, exposure, feature, io, transform


CAT_CLASS_INDEX = 3
K_VALUE = 3


def load_cifar_batch(batch_path):
    with open(batch_path, "rb") as batch_file:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="dtype\\(\\): align should be passed.*")
            batch_dict = pickle.load(batch_file, encoding="latin1")

    images = batch_dict["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    images = images.astype(np.float32) / 255.0
    labels = np.asarray(batch_dict["labels"], dtype=np.int32)
    return images, labels


def load_cifar10_dataset(dataset_dir):
    train_images = []
    train_labels = []

    for batch_number in range(1, 6):
        batch_path = os.path.join(dataset_dir, f"data_batch_{batch_number}")
        images, labels = load_cifar_batch(batch_path)
        train_images.append(images)
        train_labels.append(labels)

    test_images, test_labels = load_cifar_batch(os.path.join(dataset_dir, "test_batch"))

    train_images = np.concatenate(train_images, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    return train_images, train_labels, test_images, test_labels


def convert_to_binary_cat_labels(labels):
    return (labels == CAT_CLASS_INDEX).astype(np.int32)


def extract_image_features(images):
    feature_vectors = []

    for image in images:
        gray_image = color.rgb2gray(image).astype(np.float32)
        equalized_gray = exposure.equalize_adapthist(gray_image, clip_limit=0.03).astype(np.float32)

        full_hog = feature.hog(
            equalized_gray,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )

        upper_face = equalized_gray[:20, 4:28]
        upper_hog = feature.hog(
            upper_face,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )

        edge_map = feature.canny(equalized_gray, sigma=1.0).astype(np.float32)
        whisker_band = equalized_gray[14:28, 2:30]
        whisker_edges = feature.canny(whisker_band, sigma=0.8).astype(np.float32)
        eye_band = equalized_gray[8:18, 6:26]
        lbp = feature.local_binary_pattern(
            (equalized_gray * 255).astype(np.uint8),
            P=8,
            R=1,
            method="uniform",
        )
        lbp_histogram, _ = np.histogram(lbp, bins=np.arange(11), density=True)

        color_means = image.mean(axis=(0, 1))
        color_stds = image.std(axis=(0, 1))
        eye_features = np.array(
            [
                eye_band.mean(),
                eye_band.std(),
                eye_band.max() - eye_band.min(),
            ],
            dtype=np.float32,
        )
        whisker_features = np.concatenate(
            [
                np.array([whisker_edges.mean()], dtype=np.float32),
                whisker_edges.mean(axis=0),
            ]
        )
        edge_profiles = np.concatenate([edge_map.mean(axis=0), edge_map.mean(axis=1)])

        feature_vectors.append(
            np.concatenate(
                [
                    full_hog,
                    upper_hog,
                    edge_profiles,
                    whisker_features,
                    eye_features,
                    lbp_histogram.astype(np.float32),
                    color_means,
                    color_stds,
                ]
            )
        )

    return np.asarray(feature_vectors, dtype=np.float32)


def build_cat_focus_weights(train_features, binary_train_labels):
    cat_features = train_features[binary_train_labels == 1]
    non_cat_features = train_features[binary_train_labels == 0]

    cat_signature = cat_features.mean(axis=0)
    non_cat_signature = non_cat_features.mean(axis=0)

    weights = np.abs(cat_signature - non_cat_signature)
    weights = weights / (weights.max() + 1e-8)
    weights = 0.30 + 0.70 * weights
    return weights.astype(np.float32)


def apply_cat_focus(features, feature_weights):
    return (features * feature_weights[np.newaxis, :]).astype(np.float32)


def standardize_features(train_features, test_features):
    feature_means = train_features.mean(axis=0)
    feature_stds = train_features.std(axis=0) + 1e-6

    standardized_train_features = (train_features - feature_means) / feature_stds
    standardized_test_features = (test_features - feature_means) / feature_stds
    return standardized_train_features.astype(np.float32), standardized_test_features.astype(np.float32)


def get_standardization_parameters(train_features):
    feature_means = train_features.mean(axis=0)
    feature_stds = train_features.std(axis=0) + 1e-6
    return feature_means.astype(np.float32), feature_stds.astype(np.float32)


def apply_standardization(features, feature_means, feature_stds):
    standardized_features = (features - feature_means) / feature_stds
    return standardized_features.astype(np.float32)


def l2_normalize_rows(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    return (features / norms).astype(np.float32)


def build_cat_focused_training_subset(binary_train_labels, non_cat_ratio=2, random_seed=42):
    cat_indices = np.where(binary_train_labels == 1)[0]
    non_cat_indices = np.where(binary_train_labels == 0)[0]

    max_non_cat_count = min(non_cat_indices.size, cat_indices.size * non_cat_ratio)
    rng = np.random.default_rng(random_seed)
    sampled_non_cat_indices = rng.choice(non_cat_indices, size=max_non_cat_count, replace=False)

    subset_indices = np.concatenate([cat_indices, sampled_non_cat_indices])
    rng.shuffle(subset_indices)
    return subset_indices


def euclidean_distance_matrix(test_batch_features, train_features, train_squared_norms):
    test_squared_norms = np.sum(test_batch_features * test_batch_features, axis=1, keepdims=True)
    distances_squared = (
        test_squared_norms
        + train_squared_norms[np.newaxis, :]
        - 2.0 * test_batch_features @ train_features.T
    )
    distances_squared = np.maximum(distances_squared, 0.0)
    return np.sqrt(distances_squared, out=distances_squared)


def knn_predict(train_features, train_labels, test_features, k=K_VALUE, batch_size=100):
    train_features = np.ascontiguousarray(train_features.astype(np.float32))
    test_features = np.ascontiguousarray(test_features.astype(np.float32))
    train_labels = np.asarray(train_labels, dtype=np.int32)
    train_squared_norms = np.sum(train_features * train_features, axis=1)

    predictions = np.empty(test_features.shape[0], dtype=np.int32)
    minimum_cat_votes = (k // 2) + 1

    for start_index in range(0, test_features.shape[0], batch_size):
        end_index = min(start_index + batch_size, test_features.shape[0])
        test_batch = test_features[start_index:end_index]
        distances = euclidean_distance_matrix(test_batch, train_features, train_squared_norms)

        nearest_indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        nearest_labels = train_labels[nearest_indices]
        predictions[start_index:end_index] = (
            np.sum(nearest_labels, axis=1) >= minimum_cat_votes
        ).astype(np.int32)

        print(
            f"Processed test images {start_index + 1}-{end_index} "
            f"out of {test_features.shape[0]}"
        )

    return predictions


def knn_predict_with_details(train_features, train_labels, test_features, k=K_VALUE, batch_size=100):
    train_features = np.ascontiguousarray(train_features.astype(np.float32))
    test_features = np.ascontiguousarray(test_features.astype(np.float32))
    train_labels = np.asarray(train_labels, dtype=np.int32)
    train_squared_norms = np.sum(train_features * train_features, axis=1)

    predictions = np.empty(test_features.shape[0], dtype=np.int32)
    cat_vote_counts = np.empty(test_features.shape[0], dtype=np.int32)
    mean_neighbor_distances = np.empty(test_features.shape[0], dtype=np.float32)
    minimum_cat_votes = (k // 2) + 1

    for start_index in range(0, test_features.shape[0], batch_size):
        end_index = min(start_index + batch_size, test_features.shape[0])
        test_batch = test_features[start_index:end_index]
        distances = euclidean_distance_matrix(test_batch, train_features, train_squared_norms)

        nearest_indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        nearest_labels = train_labels[nearest_indices]
        nearest_distances = np.take_along_axis(distances, nearest_indices, axis=1)

        cat_votes = np.sum(nearest_labels, axis=1)
        cat_vote_counts[start_index:end_index] = cat_votes
        mean_neighbor_distances[start_index:end_index] = nearest_distances.mean(axis=1)
        predictions[start_index:end_index] = (cat_votes >= minimum_cat_votes).astype(np.int32)

    return predictions, cat_vote_counts, mean_neighbor_distances


def preprocess_image_for_mode(image, mode="full"):
    image = np.asarray(image)

    if image.ndim == 2:
        image = color.gray2rgb(image)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0

    height, width = image.shape[:2]

    if mode == "upper_face":
        cropped_image = image[
            : max(int(height * 0.72), 1),
            int(width * 0.12) : max(int(width * 0.88), int(width * 0.12) + 1),
        ]
    elif mode == "tight_face":
        cropped_image = image[
            : max(int(height * 0.62), 1),
            int(width * 0.18) : max(int(width * 0.82), int(width * 0.18) + 1),
        ]
    else:
        cropped_image = image

    crop_size = min(cropped_image.shape[:2])
    top = (cropped_image.shape[0] - crop_size) // 2
    left = (cropped_image.shape[1] - crop_size) // 2
    cropped_image = cropped_image[top : top + crop_size, left : left + crop_size]

    resized_image = transform.resize(
        cropped_image,
        (32, 32, 3),
        anti_aliasing=True,
        preserve_range=True,
    )
    return resized_image.astype(np.float32)


def normalize_display_image(image):
    image = np.asarray(image)

    if image.ndim == 2:
        image = color.gray2rgb(image)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return image


def load_images_from_folder(folder_path):
    image_names = []
    original_images = []
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(valid_extensions):
            image_names.append(file_name)
            original_images.append(normalize_display_image(io.imread(file_path)))

    if not original_images:
        return [], []

    return image_names, original_images


def calculate_metrics(true_labels, predicted_labels):
    true_positive = np.sum((true_labels == 1) & (predicted_labels == 1))
    true_negative = np.sum((true_labels == 0) & (predicted_labels == 0))
    false_positive = np.sum((true_labels == 0) & (predicted_labels == 1))
    false_negative = np.sum((true_labels == 1) & (predicted_labels == 0))

    accuracy = (true_positive + true_negative) / true_labels.size
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positive": int(true_positive),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
    }


def select_example_indices(true_labels, predicted_labels, max_examples=9):
    true_positive = np.where((true_labels == 1) & (predicted_labels == 1))[0]
    false_negative = np.where((true_labels == 1) & (predicted_labels == 0))[0]
    false_positive = np.where((true_labels == 0) & (predicted_labels == 1))[0]
    true_negative = np.where((true_labels == 0) & (predicted_labels == 0))[0]

    selected_indices = []

    groups = (
        true_positive[:3],
        false_negative[:3],
        false_positive[:2],
        true_negative[:2],
    )

    for group in groups:
        for index in group:
            if len(selected_indices) == max_examples:
                return selected_indices
            selected_indices.append(int(index))

    for group in (true_positive, false_negative, false_positive, true_negative):
        for index in group:
            if len(selected_indices) == max_examples:
                return selected_indices
            if int(index) not in selected_indices:
                selected_indices.append(int(index))

    return selected_indices


def plot_results(reference_cat_image, test_images, true_labels, predicted_labels, max_examples=9):
    example_indices = select_example_indices(true_labels, predicted_labels, max_examples=max_examples)

    figure, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.ravel()

    axes[0].imshow(np.clip(reference_cat_image, 0.0, 1.0))
    axes[0].set_title("Training cat reference")
    axes[0].axis("off")

    for axis, image_index in zip(axes[1:], example_indices):
        axis.imshow(test_images[image_index])
        axis.set_title(
            f"True: {'Cat' if true_labels[image_index] == 1 else 'Not cat'}\n"
            f"Pred: {'Cat' if predicted_labels[image_index] == 1 else 'Not cat'}"
        )
        axis.axis("off")

    for axis in axes[len(example_indices) + 1 :]:
        axis.axis("off")

    figure.suptitle("KNN Cat Classifier Results", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_external_classification_results(images, results):
    figure, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6))

    if len(results) == 1:
        axes = [axes]

    for axis, image, result in zip(axes, images, results):
        axis.imshow(np.clip(image, 0.0, 1.0))
        axis.set_title(
            f"{result['image_name']}\n"
            f"Pred: {result['predicted_label']}\n"
            f"Cat votes: {result['cat_votes']}/{K_VALUE * 3}\n"
            f"Views: {result['view_predictions']}"
        )
        axis.axis("off")

    figure.suptitle("Predictions For Images In cat Folder", fontsize=14)
    plt.tight_layout()
    plt.show()


def run_cat_knn_classifier(
    dataset_dir="cifar-10-batches-py",
    train_limit=None,
    test_limit=None,
    non_cat_ratio=2,
    show_plots=True,
):
    print("Loading CIFAR-10 dataset...")
    train_images, train_labels, test_images, test_labels = load_cifar10_dataset(dataset_dir)

    if train_limit is not None:
        train_images = train_images[:train_limit]
        train_labels = train_labels[:train_limit]

    if test_limit is not None:
        test_images = test_images[:test_limit]
        test_labels = test_labels[:test_limit]

    binary_train_labels = convert_to_binary_cat_labels(train_labels)
    binary_test_labels = convert_to_binary_cat_labels(test_labels)

    subset_indices = build_cat_focused_training_subset(
        binary_train_labels,
        non_cat_ratio=non_cat_ratio,
    )
    train_images = train_images[subset_indices]
    binary_train_labels = binary_train_labels[subset_indices]

    print(f"Training images used: {train_images.shape[0]}")
    print(f"Test images used: {test_images.shape[0]}")
    print(f"Cat-labelled training images: {np.sum(binary_train_labels)}")

    print("Extracting image features...")
    train_processed_images = np.asarray(
        [preprocess_image_for_mode(image, mode="full") for image in train_images],
        dtype=np.float32,
    )
    test_processed_images = np.asarray(
        [preprocess_image_for_mode(image, mode="full") for image in test_images],
        dtype=np.float32,
    )
    train_features = extract_image_features(train_processed_images)
    test_features = extract_image_features(test_processed_images)

    print("Standardising features so Euclidean distance is better balanced...")
    train_features, test_features = standardize_features(train_features, test_features)

    print("Normalising features before Euclidean-distance KNN...")
    focused_train_features = l2_normalize_rows(train_features)
    focused_test_features = l2_normalize_rows(test_features)

    print("KNN is using a cat-focused training subset for k = 3 voting.")
    subset_train_features = focused_train_features
    subset_train_labels = binary_train_labels

    print(f"Running KNN with Euclidean distance and k = {K_VALUE}...")
    predicted_labels = knn_predict(
        subset_train_features,
        subset_train_labels,
        focused_test_features,
        k=K_VALUE,
    )

    metrics = calculate_metrics(binary_test_labels, predicted_labels)
    reference_cat_image = train_processed_images[np.where(binary_train_labels == 1)[0][0]]

    print("\nEvaluation metrics")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")
    print(
        "Confusion counts -> "
        f"TP: {metrics['true_positive']}, "
        f"TN: {metrics['true_negative']}, "
        f"FP: {metrics['false_positive']}, "
        f"FN: {metrics['false_negative']}"
    )

    if show_plots:
        plot_results(reference_cat_image, test_images, binary_test_labels, predicted_labels)

    return metrics, predicted_labels


def classify_external_images(
    folder_path,
    dataset_dir="cifar-10-batches-py",
    train_limit=None,
    non_cat_ratio=1,
    show_plots=True,
):
    print("Loading CIFAR-10 dataset for external image classification...")
    train_images, train_labels, _, _ = load_cifar10_dataset(dataset_dir)

    if train_limit is not None:
        train_images = train_images[:train_limit]
        train_labels = train_labels[:train_limit]

    binary_train_labels = convert_to_binary_cat_labels(train_labels)
    subset_indices = build_cat_focused_training_subset(
        binary_train_labels,
        non_cat_ratio=non_cat_ratio,
    )
    train_images = train_images[subset_indices]
    binary_train_labels = binary_train_labels[subset_indices]

    print("Loading and preprocessing external images...")
    image_names, original_images = load_images_from_folder(folder_path)
    if len(original_images) == 0:
        print("No supported images were found in the folder.")
        return []

    view_modes = ("full", "upper_face", "tight_face")
    aggregated_predictions = np.zeros(len(image_names), dtype=np.int32)
    aggregated_cat_votes = np.zeros(len(image_names), dtype=np.int32)
    aggregated_distances = np.zeros(len(image_names), dtype=np.float32)
    per_view_labels = {image_name: [] for image_name in image_names}

    print("Extracting cat-face-aware features for multiple views of each external image...")
    for view_mode in view_modes:
        train_view_images = np.asarray(
            [preprocess_image_for_mode(image, mode=view_mode) for image in train_images],
            dtype=np.float32,
        )
        external_view_images = np.asarray(
            [preprocess_image_for_mode(io.imread(os.path.join(folder_path, name)), mode=view_mode) for name in image_names],
            dtype=np.float32,
        )

        train_features = extract_image_features(train_view_images)
        external_features = extract_image_features(external_view_images)

        feature_means, feature_stds = get_standardization_parameters(train_features)
        train_features = apply_standardization(train_features, feature_means, feature_stds)
        external_features = apply_standardization(external_features, feature_means, feature_stds)

        train_features = l2_normalize_rows(train_features)
        external_features = l2_normalize_rows(external_features)

        predictions, cat_vote_counts, mean_neighbor_distances = knn_predict_with_details(
            train_features,
            binary_train_labels,
            external_features,
            k=K_VALUE,
            batch_size=32,
        )

        aggregated_predictions += predictions
        aggregated_cat_votes += cat_vote_counts
        aggregated_distances += mean_neighbor_distances

        for image_name, prediction in zip(image_names, predictions):
            per_view_labels[image_name].append("Cat" if prediction == 1 else "Not cat")

    results = []
    for image_index, image_name in enumerate(image_names):
        final_prediction = (
            "Cat"
            if aggregated_predictions[image_index] >= 2 or aggregated_cat_votes[image_index] >= 4
            else "Not cat"
        )
        mean_distance = aggregated_distances[image_index] / len(view_modes)
        results.append(
            {
                "image_name": image_name,
                "predicted_label": final_prediction,
                "cat_votes": int(aggregated_cat_votes[image_index]),
                "mean_neighbor_distance": float(mean_distance),
                "view_predictions": ", ".join(per_view_labels[image_name]),
            }
        )
        print(
            f"{image_name} -> {final_prediction} "
            f"(view predictions: {', '.join(per_view_labels[image_name])}; "
            f"total cat votes across views: {int(aggregated_cat_votes[image_index])}/{K_VALUE * len(view_modes)})"
        )

    if show_plots:
        plot_external_classification_results(original_images, results)

    return results


def Q1_CIFAR10_Cat_KNN(
    dataset_dir="cifar-10-batches-py",
    train_limit=None,
    test_limit=None,
    non_cat_ratio=2,
    show_plots=True,
):
    return run_cat_knn_classifier(
        dataset_dir=dataset_dir,
        train_limit=train_limit,
        test_limit=test_limit,
        non_cat_ratio=non_cat_ratio,
        show_plots=show_plots,
    )


if __name__ == "__main__":
    run_cat_knn_classifier(show_plots=False)
    classify_external_images(
        folder_path="cat",
        dataset_dir="cifar-10-batches-py",
        non_cat_ratio=1,
        show_plots=True,
    )
