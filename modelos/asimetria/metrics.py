# metrics.py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot_training_history(history):
    """
    Genera gráficas de Loss y Accuracy del entrenamiento.
    Recibe el objeto history retornado por model.fit().
    """
    epochs = range(1, len(history.history['loss']) + 1)

    loss_train = history.history['loss']
    loss_val   = history.history['val_loss']
    acc_train  = history.history['accuracy']
    acc_val    = history.history['val_accuracy']

    best_epoch = int(np.argmin(loss_val)) + 1

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle('Métricas de entrenamiento — CNN Asimetría Facial', fontsize=14, y=1.02)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Gráfica 1: Loss ──────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(epochs, loss_train, color='#378ADD', linewidth=2,
             label='Train loss')
    ax1.plot(epochs, loss_val,   color='#D85A30', linewidth=2,
             linestyle='--', label='Val loss')
    ax1.axvline(x=best_epoch, color='#888780', linewidth=1,
                linestyle=':', alpha=0.8, label=f'Mejor época ({best_epoch})')
    ax1.set_title('Pérdida (Binary Crossentropy)', fontsize=12)
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend(framealpha=0.6, fontsize=10)
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.spines[['top', 'right']].set_visible(False)

    # ── Gráfica 2: Accuracy ──────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(epochs, acc_train, color='#378ADD', linewidth=2,
             label='Train accuracy')
    ax2.plot(epochs, acc_val,   color='#D85A30', linewidth=2,
             linestyle='--', label='Val accuracy')
    ax2.axvline(x=best_epoch, color='#888780', linewidth=1,
                linestyle=':', alpha=0.8, label=f'Mejor época ({best_epoch})')
    ax2.set_title('Exactitud (Accuracy)', fontsize=12)
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.legend(framealpha=0.6, fontsize=10)
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gráfica guardada en training_metrics.png")


def print_summary(history):
    """Imprime un resumen numérico del entrenamiento."""
    best = int(np.argmin(history.history['val_loss']))
    print("\n── Resumen ──────────────────────────────")
    print(f"  Mejor época   : {best + 1}")
    print(f"  Val loss      : {history.history['val_loss'][best]:.4f}")
    print(f"  Val accuracy  : {history.history['val_accuracy'][best]:.4f}")
    if 'val_auc' in history.history:
        print(f"  Val AUC       : {history.history['val_auc'][best]:.4f}")
    if 'val_recall' in history.history:
        print(f"  Val recall    : {history.history['val_recall'][best]:.4f}")
    print("─────────────────────────────────────────")