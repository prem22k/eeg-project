from models.eegnet import EEGNet
import data_preprocessing as dp
import numpy as np
import tensorflow as tf
import sklearn
import os
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def _save_kfold_evaluation_artifacts(
    y_true,
    y_pred,
    y_scores,
    fold_accuracies,
    output_dir='outputs',
    output_prefix='kfold',
    class_names=None,
):
    """Save publication-style evaluation artifacts after k-fold evaluation."""
    if class_names is None:
        class_names = ['Up', 'Down', 'Left', 'Right']

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_scores = np.asarray(y_scores)
    n_classes = len(class_names)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_prefix = f'{output_prefix}_{timestamp}'

    confusion_path = os.path.join(output_dir, f'{run_prefix}_confusion_matrix.png')
    roc_path = os.path.join(output_dir, f'{run_prefix}_roc_curve.png')
    report_path = os.path.join(output_dir, f'{run_prefix}_classification_report.txt')

    # High-resolution confusion matrix for publication use.
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    ax.set_title('Confusion Matrix (All K-Fold Predictions)')
    fig.tight_layout()
    fig.savefig(confusion_path, dpi=300)
    plt.close(fig)

    # One-vs-rest multiclass ROC curves over concatenated fold predictions.
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    roc_plotted = False
    for class_index, class_name in enumerate(class_names):
        if np.unique(y_true_bin[:, class_index]).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, class_index], y_scores[:, class_index])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
        roc_plotted = True

    ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', label='Chance')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multiclass ROC (One-vs-Rest)')
    if roc_plotted:
        ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(roc_path, dpi=300)
    plt.close(fig)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=np.arange(n_classes),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    mean_accuracy = float(np.mean(fold_accuracies)) if fold_accuracies else float(np.mean(y_true == y_pred))
    std_accuracy = float(np.std(fold_accuracies)) if fold_accuracies else 0.0
    with open(report_path, 'w') as report_file:
        report_file.write('K-Fold Classification Report\n')
        report_file.write('============================\n\n')
        report_file.write(report_text)
        report_file.write('\n')
        report_file.write(f'Mean Accuracy Across Folds: {mean_accuracy:.4f}\n')
        report_file.write(f'Std Accuracy Across Folds: {std_accuracy:.4f}\n')
        report_file.write(f'Fold Accuracies: {np.array(fold_accuracies).tolist()}\n')

    print(f'Saved confusion matrix: {confusion_path}')
    print(f'Saved ROC curve: {roc_path}')
    print(f'Saved classification report: {report_path}')


def augment_pipe(data, events, noise):
    aug_data = data + np.random.normal(0, np.random.rand() * 2, data.shape) # 100_000
    for i in range(aug_data.shape[0]):
        if np.random.rand() < 0.5: aug_data[i] = np.fliplr(aug_data[i])
        if np.random.rand() < 0.5: aug_data[i] = np.flipud(aug_data[i])
        # salt pepper
        p = np.random.rand() * 0.4  # 0.4
        r = np.random.rand(*aug_data[i].shape)
        u, l = r > (1 - p/2), r < p/2
        aug_data[i][u] = 1#np.max(aug_data[i])
        aug_data[i][l] = -1#np.min(aug_data[i])
    
    aug_data += 20 * noise.reshape((*noise.shape, 1))[:data.shape[0]]
    
    return aug_data, events


def kfold_training(data, labels, model_provided, batch_size, epochs, k=4):
    """K-Fold train and test a model on data and labels.

    :param data: models input data
    :type data: numpy array
    :param labels: labels corresponding to data
    :type labels: numpy array
    :param model_path: path of the saved model to be used (pretrained or not)
    :type model_path: string
    :param batch_size: datasets batch size
    :type batch_size: integer
    :param epochs: number of epochs for which to train the model
    :type epochs: integer
    :param k: number of folds the data should be split into when training and
        testing, defaults to 4
    :type k: integer, optional
    :return: the history of the k folds
    :rtype: list containing dictonaries saving the different metrics
    """
    # shuffle data and labels
    data, labels = sklearn.utils.shuffle(data, labels)
    # create k data and label splits
    X = []
    Y = []
    for i in range(k):
        n = data.shape[0]
        X.append(data[int(n/k * i):int(n/k * i + n/k)])
        Y.append(labels[int(n/k * i):int(n/k * i + n/k)])
    # list accumulating every folds metrics 
    k_history = []
    y_true_all = []
    y_pred_all = []
    y_scores_all = []
    fold_accuracies = []
    for k_i in range(k):
        print(f"{k_i+1} of {k} starting...")
        tf.keras.backend.clear_session()
        # concat k-1 splits
        X_train = np.concatenate([d for j, d in enumerate(X) if j != k_i])
        Y_train = np.concatenate([d for j, d in enumerate(Y) if j != k_i])
        X_test = X[k_i]
        Y_test = Y[k_i]
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        # train dataset
        dataset_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).with_options(options)
        dataset_train = dp.preprocessing_pipeline(dataset_train, batch_size=batch_size)
        # Keep evaluation deterministic (no shuffle) so y_true aligns with predictions.
        dataset_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).with_options(options)
        dataset_test = dataset_test.cache().batch(batch_size).prefetch(2)
        tf.debugging.set_log_device_placement(True)
        mirrored_strategy = tf.distribute.MirroredStrategy()
        
        with mirrored_strategy.scope():
            # load pretrained model
            model = tf.keras.models.load_model(model_provided) if type(model_provided) is str else tf.keras.models.clone_model(model_provided)
        # fit model to k-folded data
        hist = model.fit(dataset_train, epochs=epochs, verbose=1, validation_data=dataset_test)

        # Collect fold predictions for aggregate post-kfold evaluation visuals.
        y_score_fold = model.predict(dataset_test, verbose=0)
        y_true_fold = np.argmax(Y_test, axis=1) if Y_test.ndim > 1 else Y_test.astype(int)
        y_pred_fold = np.argmax(y_score_fold, axis=1)
        y_true_all.append(y_true_fold)
        y_pred_all.append(y_pred_fold)
        y_scores_all.append(y_score_fold)
        fold_accuracies.append(float(np.mean(y_true_fold == y_pred_fold)))

        del model
        # add metric history to accumulator
        k_history.append(hist.history)

    if y_true_all:
        _save_kfold_evaluation_artifacts(
            np.concatenate(y_true_all, axis=0),
            np.concatenate(y_pred_all, axis=0),
            np.concatenate(y_scores_all, axis=0),
            fold_accuracies,
        )
    return k_history


def pretrain_tester(pretrain_dataset, pretrain_val_dataset,
                   train_data, train_events, n_checks, pretrain_epochs, epochs,
                   batch_size, freeze_layers, dropout, kernel_length):
    """Pretrain and 'post-pre-train' a model.

    :param pretrain_dataset: dataset to pretrain the model with
    :type pretrain_dataset: tf.data.Dataset
    """
    ###### PRETRAIN AND TRANSFER LEARNING N_CHECKS TIMES
    pretrain_history_accumulator = []
    train_history_accumulator = []
    for n in range(n_checks):
        print(f"{n} of {n_checks}!")
        ###### PRETRAIN MODEL
        print("Pretraining...")
        tf.debugging.set_log_device_placement(True)
        # tensorflows mirrored strategy adds support to do synchronous distributed
        # training on multiple GPU's
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            # create EEGNet (source: https://github.com/vlawhern/arl-eegmodels)
            model_pretrain = EEGNet(nb_classes=4, Chans=train_data.shape[1],
                                    Samples=train_data.shape[2], dropoutRate=dropout,
                                    kernLength=kernel_length, F1=8, D=2, F2=16, dropoutType='Dropout')
            # adam optimizer
            optimizer = tf.keras.optimizers.Adam()
        # compile model
        model_pretrain.compile(loss='categorical_crossentropy',
                                optimizer=optimizer,
                                metrics=['accuracy'])
        # fit model to pretrain data
        pretrain_history = model_pretrain.fit(pretrain_dataset, epochs=pretrain_epochs,
                                            verbose=1, validation_data=pretrain_val_dataset)
        # append pretrain history to accumulator
        pretrain_history_accumulator.append(pretrain_history.history)
        # save pretrained model so it can be used for transfer learning
        path = './models/saved_models/pretrained_model01.keras'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("FREEZE?")
        for freeze_index in freeze_layers:
            # function to get trainable parameters
            trainable_params = lambda: np.sum([np.prod(v.shape) for v in model_pretrain.trainable_weights])
            print("trainable parameters before freezing:", trainable_params())
            model_pretrain.layers[freeze_index].trainable = False
            print("after:", trainable_params())
        model_pretrain.save(path)
        del model_pretrain
        print("Pretraining Done")
        # TRANSFER LEARNING
        # kfold testing of transfer learning
        k_history = kfold_training(train_data, train_events, path, batch_size, epochs)
        # add kfold metric-history
        train_history_accumulator.append(k_history)
        print("\n\nN: ", n, "     ######################\n")
        print("Mean for K Folds:", np.mean([h['val_accuracy'][-1] for h in k_history]))
        print("New Total Mean:", np.mean([h['val_accuracy'][-1] for h in np.concatenate(train_history_accumulator)]))
    return pretrain_history_accumulator, train_history_accumulator