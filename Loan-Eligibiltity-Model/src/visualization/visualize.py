import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import os

def LoanStatus_barchart(dataset):
    dataset['Loan_Status'].value_counts().plot.bar()
    plt.savefig('reports/figures/loan_status_barchart.png')
    plt.show()

def LoanAmount_displot(dataset):
    sns.distplot(dataset['LoanAmount'])
    plt.savefig('reports/figures/LoanAmount_displot.png')
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