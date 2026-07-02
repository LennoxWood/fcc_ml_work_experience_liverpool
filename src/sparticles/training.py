"""
training.py

Training loop, evaluation, and results plotting for the GAT classifier.
Import these functions in the notebook rather than defining them there.

Usage:
    from training import train, plot_training_curves, plot_results, plot_significance
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics import Accuracy
from sklearn.metrics import roc_curve, auc


def train(model, train_loader, test_loader, optimizer, scheduler,
          n_epochs, sig_weight, device, save_dir='Saved Models'):
    """
    Run the full training loop.

    Parameters
    ----------
    model        : GAT instance (already moved to device)
    train_loader : DataLoader for training set
    test_loader  : DataLoader for test set
    optimizer    : torch optimizer
    scheduler    : learning rate scheduler
    n_epochs     : number of epochs to train for
    sig_weight   : positive class weight for binary cross-entropy
                   (set to ttb_no/sig_no to correct for class imbalance)
    device       : torch.device
    save_dir     : where to save the final model checkpoint

    Returns
    -------
    dict with keys: train_losses, test_losses, train_accuracies,
                    test_accuracies, epoch_losses, epoch_accs
    """
    compute_acc = Accuracy(task='binary').to(device)
    pos_weight  = torch.tensor(sig_weight, device=device)

    train_losses, test_losses         = [], []
    train_accuracies, test_accuracies = [], []
    epoch_losses, epoch_accs          = [], []

    os.makedirs(save_dir, exist_ok=True)
    start = time.time()

    for epoch in range(n_epochs):

        # ── Training ─────────────────────────────────────────────────────
        model.train()
        last_train_loss, last_train_acc = None, None

        for batch in tqdm(train_loader, leave=False,
                          desc=f'Epoch {epoch}/{n_epochs-1} train'):
            batch = batch.to(device)
            optimizer.zero_grad()
            out  = model(batch.data_norm.float(), batch.edge_index, batch.batch)
            loss = F.binary_cross_entropy_with_logits(
                out.squeeze(), batch.y.float(), pos_weight=pos_weight
            )
            loss.backward()
            train_losses.append(loss.detach().item())
            train_accuracies.append(
                compute_acc(out.squeeze(), batch.y.float()).cpu().detach()
            )
            optimizer.step()
            scheduler.step()
            last_train_loss = loss.detach().item()
            last_train_acc  = train_accuracies[-1]

        # ── Validation ───────────────────────────────────────────────────
        model.eval()
        last_test_loss, last_test_acc = None, None

        with torch.no_grad():
            for batch in tqdm(test_loader, leave=False,
                              desc=f'Epoch {epoch}/{n_epochs-1} val'):
                batch = batch.to(device)
                out  = model(batch.data_norm.float(), batch.edge_index, batch.batch)
                loss = F.binary_cross_entropy_with_logits(
                    out.squeeze(), batch.y.float(), pos_weight=pos_weight
                )
                test_losses.append(loss.detach().item())
                test_accuracies.append(
                    compute_acc(out.detach().squeeze(), batch.y.float()).cpu().detach()
                )
                last_test_loss = loss.detach().item()
                last_test_acc  = test_accuracies[-1]

        epoch_losses.append(last_test_loss)
        epoch_accs.append(last_test_acc * 100)

        elapsed = (time.time() - start) / 60
        print(f'Epoch {epoch:>3} | '
              f'Train loss: {last_train_loss:.3f}  acc: {last_train_acc*100:.1f}% | '
              f'Test  loss: {last_test_loss:.3f}  acc: {last_test_acc*100:.1f}% | '
              f'Time: {elapsed:.1f} min')

    # Save final checkpoint
    torch.save({
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses':         train_losses,
        'test_losses':          test_losses,
        'train_accuracies':     train_accuracies,
        'test_accuracies':      test_accuracies,
    }, os.path.join(save_dir, f'GAT_{n_epochs}epochs.pth'))
    print(f'\nModel saved to {save_dir}/GAT_{n_epochs}epochs.pth')

    return dict(
        train_losses=train_losses,   test_losses=test_losses,
        train_accuracies=train_accuracies, test_accuracies=test_accuracies,
        epoch_losses=epoch_losses,   epoch_accs=epoch_accs,
    )


def get_predictions(model, test_loader):
    """
    Run the model over the test set and return true labels and predicted
    probabilities as numpy arrays.
    """
    model.cpu()
    model.eval()

    true_labels      = torch.cat([batch.y for batch in test_loader.dataset]).numpy()
    predicted_logits = torch.cat([
        model(batch.data_norm.float(), batch.edge_index, batch.batch)
              .detach().squeeze()
        for batch in test_loader
    ]).numpy()
    predicted_probs  = torch.sigmoid(torch.tensor(predicted_logits)).numpy()

    return true_labels, predicted_probs


def plot_training_curves(history, save_dir='Saved Figures/GNN', show=True):
    """Plot loss and accuracy curves from the dict returned by train()."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['epoch_losses'], color='steelblue', linewidth=2)
    ax1.set_xlabel('Epoch');  ax1.set_ylabel('Loss')
    ax1.set_title('Test loss per epoch')

    ax2.plot(history['epoch_accs'], color='darkorange', linewidth=2)
    ax2.set_xlabel('Epoch');  ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Test accuracy per epoch')

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, 'training_curves.png'))
    fig.savefig(os.path.join(save_dir, 'training_curves.pdf'))
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_roc_curve(true_labels, predicted_probs,
                   save_dir='Saved Figures/GNN', show=True):
    """Plot the ROC curve and return the AUC value."""
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc     = auc(fpr, tpr)

    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'GNN  (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2,
             linestyle='--', label='Random guess')
    plt.xlabel('Background efficiency (FPR)', fontsize=14)
    plt.ylabel('Signal efficiency (TPR)',      fontsize=14)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(fontsize=13)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.savefig(os.path.join(save_dir, 'roc_curve.pdf'))
    if show:
        plt.show()
    else:
        plt.close()

    return roc_auc


def plot_output_distribution(true_labels, predicted_probs,
                              save_dir='Saved Figures/GNN', show=True):
    """Plot the GNN score distribution for signal and background."""
    signal_probs     = predicted_probs[true_labels == 1]
    background_probs = predicted_probs[true_labels == 0]

    bins = np.linspace(0, 1, 101)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    for ax, yscale in zip([ax1, ax2], ['linear', 'log']):
        ax.hist(signal_probs,     bins=bins, alpha=0.7,
                label='Signal',     color='#ea6aea', density=True)
        ax.hist(background_probs, bins=bins, alpha=0.7,
                label='Background', color='#93f393', density=True)
        ax.set_yscale(yscale)
        ax.set_ylabel('Event Density')
        ax.legend()

    ax2.set_xlabel('GNN Score')
    fig.suptitle('GNN Output Distribution', fontsize=14)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, 'output_distribution.png'))
    fig.savefig(os.path.join(save_dir, 'output_distribution.pdf'))
    if show:
        plt.show()
    else:
        plt.close(fig)

    return signal_probs, background_probs


def plot_significance(signal_probs, background_probs,
                      signal_scale, background_scale,
                      max_signal=None, max_background=None,
                      save_dir='Saved Figures/GNN', show=True):
    """
    Calculate and plot cumulative and per-bin S/√B significance vs GNN score.

    Parameters
    ----------
    signal_probs, background_probs : arrays of GNN scores from get_predictions()
    signal_scale, background_scale : physical event weights (from analysis.py)
    max_signal, max_background     : total events available in the full dataset
        file (from analysis.MAX_SIGNAL_EVENTS / MAX_BACKGROUND_EVENTS). When
        provided, the significance uses the full physical yield scaled by the
        GNN cut efficiency, so the result is independent of how many events
        were loaded. When None, falls back to using the sample size directly.

    Returns
    -------
    float : peak cumulative significance
    """
    n_sig = max_signal     if max_signal     is not None else len(signal_probs)
    n_bkg = max_background if max_background is not None else len(background_probs)
    bins = np.linspace(0, 1, 101)
    # Weight each event so that all events together sum to the full physical
    # yield, scaled by the GNN score distribution from the sample.
    sig_w_per_event = signal_scale     * n_sig / len(signal_probs)
    bkg_w_per_event = background_scale * n_bkg / len(background_probs)
    ns   = np.histogram(signal_probs,     bins=bins,
                        weights=np.full(len(signal_probs),     sig_w_per_event))[0]
    nb   = np.histogram(background_probs, bins=bins,
                        weights=np.full(len(background_probs), bkg_w_per_event))[0]

    # Cumulative significance scanning from right (high score → signal-like)
    signifs, cumul = [], 0.0
    for si, bi in zip(reversed(ns), reversed(nb)):
        if bi > 0:
            cumul += (si / np.sqrt(bi)) ** 2
        signifs.append(np.sqrt(cumul))
    signifs  = list(reversed(signifs))
    bincents = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]

    best_score  = bincents[np.argmax(signifs)]
    best_signif = max(signifs)
    no_cut      = (signal_scale * n_sig) / np.sqrt(background_scale * n_bkg)

    print(f'Significance with no GNN cut:  {no_cut:.2f}')
    print(f'Best cumulative significance:  {best_signif:.2f}  '
          f'(at GNN score > {best_score:.2f})')
    if best_signif >= 5:
        print('That is above 5σ — discovery territory!')
    elif best_signif >= 3:
        print('That is above 3σ — evidence level!')
    else:
        print('Below 3σ — the model may need more training or data.')

    per_bin = [si / np.sqrt(bi) if bi > 0 else 0 for si, bi in zip(ns, nb)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(bincents, signifs, color='blueviolet', linewidth=2)
    ax1.axvline(best_score, color='red',    linestyle='--',
                label=f'Best cut ({best_score:.2f})')
    ax1.axhline(3,          color='orange', linestyle=':',  label='3σ evidence')
    ax1.axhline(5,          color='red',    linestyle=':',  label='5σ discovery')
    ax1.set_xlabel('GNN Score threshold')
    ax1.set_ylabel('Cumulative S/√B')
    ax1.set_title('Cumulative significance vs GNN score')
    ax1.legend();  ax1.grid(True, alpha=0.3)

    ax2.plot(bincents, per_bin, color='steelblue', linewidth=2)
    ax2.set_xlabel('GNN Score')
    ax2.set_ylabel('S/√B per bin')
    ax2.set_title('Per-bin significance vs GNN score')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, 'significance.png'))
    fig.savefig(os.path.join(save_dir, 'significance.pdf'))
    if show:
        plt.show()
    else:
        plt.close(fig)

    return best_signif