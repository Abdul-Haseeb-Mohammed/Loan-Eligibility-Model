import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import os

def LoanStatus_barchart(dataset, save_path=None):
    dataset['Loan_Status'].value_counts().plot.bar()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Plot saved to {os.path.abspath(save_path)}")
    else:
        plt.show()

def LoanAmount_displot(dataset, save_path=None):
    sns.distplot(dataset['LoanAmount'])
    plt.close()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Plot saved to {os.path.abspath(save_path)}")
    else:
        plt.show()


def plot_tree(model, feature_names, save_path=None):
    plt.figure(figsize=(20, 10))
    tree.plot_tree(model, feature_names=feature_names, filled=True, fontsize=10)
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Decision tree plot saved to {os.path.abspath(save_path)}")
    else:
        plt.show()