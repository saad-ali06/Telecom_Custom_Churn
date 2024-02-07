import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,  classification_report, precision_recall_curve, auc


def customers_used_various_services(df):
    """
    The function `customers_used_various_services` creates a subplot of bar charts to visualize the
    count of customers who used various services.
    
    :param df: The parameter `df` is a pandas DataFrame that contains the data you want to analyze
    """
    
    services = ['PhoneService','MultipleLines','InternetService','OnlineSecurity',
           'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']

    fig, axes = plt.subplots(nrows = 3,ncols = 3,figsize = (15,12))
    color = ["#25404d", "#68ba8a", "#4d3225"]
    for i, item in enumerate(services):
        if i < 3:
            ax = df[item].value_counts().plot(kind = 'bar',ax=axes[i,0],rot = 0, color=color)
            
        elif i >=3 and i < 6:
            ax = df[item].value_counts().plot(kind = 'bar',ax=axes[i-3,1],rot = 0, color=color)
            
        elif i < 9:
            ax = df[item].value_counts().plot(kind = 'bar',ax=axes[i-6,2],rot = 0, color=color)
        ax.set_title(item)
        
        
def distribution_of_dataset(df, name='Churn'):
    """
    The function `distribution_of_dataset` creates a count plot to visualize the distribution of a
    specific variable in a dataset.
    
    :param df: The parameter "df" is the dataset that you want to analyze. It should be a pandas
    DataFrame object containing the data you want to visualize
    :param name: The `name` parameter is used to specify the column name in the dataset that represents
    the variable for which you want to visualize the distribution. In this case, it is set to 'Churn',
    indicating that you want to visualize the distribution of the 'Churn' variable in the dataset,
    defaults to Churn (optional)
    """
   
    # Create the count plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.countplot(x=name, data=df, palette="Set2")  # Adjust the palette for color preference

    plt.title('Churn Distribution (Count)')
    plt.xlabel('Churn Status')
    plt.ylabel('# Customers')
    plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.legend()
    plt.show()
    
def distribution_of_gender(df, name='gender'):
    colors = ["#25404d", "#68ba8a"]
    ax = (df[name].value_counts()*100.0 /len(df)).plot(kind='bar', stacked = True, rot = 0, color = colors)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel('% Customers')
    ax.set_xlabel('Gender')
    ax.set_ylabel('% Customers')
    ax.set_title('Gender Distribution')

    # create a list to collect the plt.patches data
    totals = []

    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_width())

    # set individual bar lables using above list
    total = sum(totals)

    for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(i.get_x()+.15, i.get_height()-3.5, \
                str(round((i.get_height()/total), 1))+'%',
                fontsize=12,
                color='white',
            weight = 'bold')
        
        
def distribution_of_age_parter_dependent(df):
    """
    The function `distribution_of_age_parter_dependent` creates a subplot grid with 2 rows and 2 columns
    and plots various distribution charts related to age, partner, and dependents using the provided
    DataFrame.
    
    :param df: The parameter `df` is a DataFrame that contains the data for analysis. It is assumed to
    have columns named 'SeniorCitizen', 'Dependents', 'Partner', and 'customerID'
    """
    # The code block you provided is creating a subplot grid with 2 rows and 2 columns using
    # `plt.subplots(2, 2, figsize=(12, 8))`.
    # Assuming you have your 'df' DataFrame loaded

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # Create a 2x2 subplot grid

    # Plot 1: Pie chart for % of Senior Citizens
    ax = axes[0, 0]
    (df['SeniorCitizen'].value_counts() * 100.0 / len(df)).plot.pie(
        autopct="%.1f%%",
        labels=["No", "Yes"],
        ax=ax,
        fontsize=12,
    )
    ax.set_ylabel("Senior Citizens", fontsize=12)
    ax.set_title("% of Senior Citizens", fontsize=12)

    # Plot 2: Stacked bar chart for % Customers with dependents and partners
    df2 = pd.melt(df, id_vars=["customerID"], value_vars=["Dependents", "Partner"])
    df3 = df2.groupby(["variable", "value"]).count().unstack() * 100 / len(df)
    colors = ["#25404d", "#cef2f2"]
    ax = axes[0, 1]
    df3.loc[:, "customerID"].plot.bar(
        stacked=True,
        color=colors,
        ax=ax,
        rot=0,
        width=0.2,
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel("% Customers", size=14)
    ax.set_xlabel("")
    ax.set_title("% Customers with dependents and partners", size=14)
    ax.legend(loc="center", prop={"size": 14})
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.annotate(
            "{:.0f}%".format(height),
            (p.get_x() + 0.25 * width, p.get_y() + 0.4 * height),
            color="white",
            weight="bold",
            size=14,
        )

    # Plot 3: Stacked bar chart for % Customers with/without dependents based on partner
    partner_dependents = df.groupby(["Partner", "Dependents"]).size().unstack()
    ax = axes[1, 0]
    (partner_dependents.T * 100.0 / partner_dependents.T.sum()).T.plot(
        kind="bar",
        width=0.2,
        stacked=True,
        rot=0,
        ax=ax,
        color=colors,
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(
        loc="center", prop={"size": 14}, title="Dependents", fontsize=14
    )
    ax.set_ylabel("% Customers", size=14)
    ax.set_title(
        "% Customers with/without dependents based on whether they have a partner", size=14
    )
    ax.xaxis.label.set_size(14)
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.annotate(
            "{:.0f}%".format(height),
            (p.get_x() + 0.25 * width, p.get_y() + 0.4 * height),
            color="white",
            weight="bold",
            size=14,
        )

    # Adjust layout and spacing
    plt.tight_layout()
    plt.show()
    
    
def distribution_of_contract_tenure(df):
    # Assuming you have your 'df' DataFrame loaded

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # Create a 2x2 subplot grid

    # Plot 1: Bar chart for # of Customers by Contract Type
    ax = axes[0, 0]
    df['Contract'].value_counts().plot(kind='bar', rot=0, width=0.3, ax=ax)
    ax.set_ylabel('# of Customers')
    ax.set_title('# of Customers by Contract Type')

    # Plot 2: Distribution of tenure for "Month-to-month" contracts
    ax = axes[0, 1]
    sns.distplot(df[df['Contract'] == 'Month-to-month']['tenure'],
                hist=True, kde=False,
                bins=int(180/5), color='steelblue',
                hist_kws={'edgecolor':'blue'},
                kde_kws={'linewidth': 4},
                ax=ax)
    ax.set_ylabel('# of Customers')
    ax.set_xlabel('Tenure (months)')
    ax.set_title('Month to Month Contract')

    # Plot 3: Distribution of tenure for "One year" contracts
    ax = axes[1, 0]
    sns.distplot(df[df['Contract'] == 'One year']['tenure'],
                hist=True, kde=False,
                bins=int(180/5), color='darkblue',
                hist_kws={'edgecolor':'blue'},
                kde_kws={'linewidth': 4},
                ax=ax)
    ax.set_xlabel('Tenure (months)', size=14)
    ax.set_title('One Year Contract', size=14)

    # Plot 4: Distribution of tenure for "Two year" contracts
    ax = axes[1, 1]
    sns.distplot(df[df['Contract'] == 'Two year']['tenure'],
                hist=True, kde=False,
                bins=int(180/5), color='#76b0b0',
                hist_kws={'edgecolor':'blue'},
                kde_kws={'linewidth': 4},
                ax=ax)
    ax.set_xlabel('Tenure (months)')
    ax.set_title('Two Year Contract')

    # Adjust layout and spacing
    plt.tight_layout()
    plt.show()
    
    
def correlation_graph(df):
    # Calculate correlations
    correlations = df.corr()

    # Create the correlation heatmap
    plt.figure(figsize=(18, 15))
    sns.heatmap(correlations,
                annot=True,
                cmap='coolwarm',
                vmin=-1, vmax=1,
                linewidths=0.5,
                fmt='.2f')

    # Customize label colors
    labels = plt.gca().get_xticklabels() + plt.gca().get_yticklabels()
    for label in labels:
        label.set_color('black')

    # Enhance clarity with title and labels
    plt.title('Correlation Matrix', fontsize=16, color='black')
    plt.xlabel('Variables', fontsize=14, color='black')
    plt.ylabel('Variables', fontsize=14, color='black')

    # Rotate x-axis labels for readability
    plt.xticks(rotation=90)

    # Add colorbar to visualize correlation values
    # cbar = plt.colorbar(label='Correlation')  # Add label for clarity
    # cbar.ax.tick_params(labelsize=12)  # Adjust colorbar label size

    plt.show()
    
    
def correlation_bar_graph(df_dummies):
    """
    The function `correlation_bar_graph` creates a bar graph to visualize the correlation between the
    'Churn' column and other columns in a DataFrame.
    
    :param df_dummies: The parameter `df_dummies` is a pandas DataFrame that contains the data for which
    you want to calculate the correlation and create a bar graph. It is assumed that the DataFrame has
    been preprocessed and contains only numerical columns or columns that have been converted to
    numerical values using one-hot encoding or similar techniques
    """
    plt.figure(figsize=(15, 8))
    correlations = df_dummies.corr()['Churn'].sort_values(ascending=False)

    # Choose a colormap for the bars (e.g., 'viridis', 'coolwarm', 'plasma', etc.)
    cmap = plt.cm.get_cmap('plasma')

    # Create a normalized index for mapping colors to bars
    norm = plt.Normalize(correlations.min(), correlations.max())

    # Plot the correlations with colors based on the colormap
    correlations.plot(kind='bar', color=[cmap(norm(val)) for val in correlations])

    plt.show()
    

def conf_matrix(y_test, y_pred):
    """
    The `conf_matrix` function calculates and displays the confusion matrix, accuracy, classification
    report, and AUC-ROC score for a given set of true labels and predicted labels.
    
    :param y_test: y_test is the true labels or target values for the test set. It is a 1-dimensional
    array or list containing the actual labels of the test data
    :param y_pred: The predicted labels for the test data
    """
    # Create a confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Display the confusion matrix using a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # Display classification report
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    
    print("AUC-ROC Score:")
    print(roc_auc_score(y_test, y_pred))
    print("\n")
    
    
def evaluate_at_threshold(y_test, y_prob, threshold):
    """
    The function evaluates the performance of a binary classification model at a given threshold by
    comparing the predicted labels with the true labels.
    
    :param y_test: The parameter `y_test` is the true labels or target values of the test set. It is a
    1-dimensional array or list containing the actual labels for each sample in the test set
    :param y_prob: The parameter `y_prob` is a numpy array that contains the predicted probabilities for
    each class
    :param threshold: The threshold is a value between 0 and 1 that is used to classify the predicted
    probabilities. If the predicted probability is greater than or equal to the threshold, it is
    classified as 1, otherwise it is classified as 0
    """
    y_pred = (y_prob >= threshold).astype(int)
    print(f"Threshold: {threshold}")
    conf_matrix(y_test, y_pred)
    
def p_r_curve(y_test, y_proba):
    """
    The function calculates the precision-recall curve, plots it, calculates the area under the curve,
    and finds the optimal threshold that maximizes the F1-score.
    
    :param y_test: The true labels of the test set. It is a binary array or list where each element
    represents the true class label of a sample in the test set
    :param y_proba: The predicted probabilities of the positive class
    :return: the optimal threshold that maximizes the F1-score.
    """
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    # Calculate the area under the precision-recall curve
    auc_score = auc(recall, precision)

    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AUC = {auc_score:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    # Find the optimal threshold that maximizes F1-score
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    return optimal_threshold