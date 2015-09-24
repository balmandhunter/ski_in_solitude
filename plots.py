import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from sklearn.learning_curve import learning_curve


def plot_params():
    size = 18
    a = plt.rc('xtick', labelsize = size)
    b = plt.rc('ytick', labelsize = size)
    return a, b, plt.gca(), size

def plot_ticket_sales(df):
    plt.figure(facecolor='w', figsize = (15,20))
    a, b, axes, label_size = plot_params()

    ax1 = plt.subplot(311)
    df['tickets'].plot(marker = '.',linestyle = '-', label = 'Total Tickets Sold')
    plt.ylabel('Number of Tickets', size = label_size)
    plt.legend(fontsize = label_size)
    plt.xlabel('Date', size = label_size)

    ax2 = plt.subplot(312)
    df['is_CO'].plot(marker = '.',linestyle = '-', label = 'Sales to CO Residents')
    plt.ylabel('Number of Tickets', size = label_size)
    plt.legend(fontsize = label_size)
    plt.xlabel('Date', size = label_size)

    ax3 = plt.subplot(313)
    df['not_CO'].plot(marker = '.',linestyle = '-', label = 'Sales to Out-of-State Visitors')
    plt.ylabel('Number of Tickets', size = label_size)
    plt.legend(fontsize = label_size)
    plt.xlabel('Date', size = label_size)


def make_correlation_plot(df):
    f, ax = plt.subplots(figsize=(12, 12))
    sns.corrplot(df, annot=True, sig_stars=False,
             diag_names=False, ax=ax)


def plot_two_lines(df, col1, col2, y_title1, y_title2, xlim):
	plt.figure(facecolor='w', figsize = (15,10))
	a, b, axes, label_size = plot_params()

	df[col1].plot(marker = 'o',linestyle = '-', label = y_title1, xlim = xlim)
	df[col2].plot(marker = 'o',linestyle = '-', label = y_title2, xlim = xlim)
	plt.legend(fontsize = label_size)


def plot_one_line(df, col1, y_title1, xlim):
	plt.figure(facecolor='w', figsize = (15,10))
	a, b, axes, label_size = plot_params()

	df[col1].plot(marker = 'o',linestyle = '-', label = y_title1, xlim = xlim)
	plt.legend(fontsize = label_size)


def plot_six_lines(df, col1, col2, col3, col4, col5, col6, y_title1, y_title2, y_title3, y_title4, y_title5, y_title6, xlim):
    plt.figure(facecolor='w', figsize = (15,10))
    a, b, axes, label_size = plot_params()

    df[col1].plot(marker = 'o',linestyle = '-', label = y_title1, xlim = xlim)
    df[col2].plot(marker = 'o',linestyle = '-', label = y_title2, xlim = xlim)
    df[col3].plot(marker = 'o',linestyle = '-', label = y_title3, xlim = xlim)
    df[col4].plot(marker = 'o',linestyle = '-', label = y_title4, xlim = xlim)
    df[col5].plot(marker = 'o',linestyle = '-', label = y_title5, xlim = xlim)
    df[col6].plot(marker = 'o',linestyle = '-', label = y_title6, xlim = xlim)
    plt.legend(fontsize = label_size)


def plot_hist(values, title, ):
    plt.figure(figsize = (10,5), facecolor='w')
    a, b, axes, label_size = plot_params()
    h = sorted(values)
    fit = stats.norm.pdf(h, np.mean(h), np.std(h))
    plt.plot(h, fit, '-o')
    plt.title(title, size = label_size)
    min1 = min(values)
    max1 = max(values)
    plt.hist(h)
    plt.show()


def make_bar_chart(crowd, y_label, dates):
    a, b, label_size = plot_params()
    N = len(crowd)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35      # the width of the bars

    fig, ax = plt.subplots(figsize=(20, 10), facecolor='white', frameon=False)
    #plt.box(on='off')
    rects1 = ax.bar(ind, crowd, width, color='#bdbdbd', edgecolor = "none")

    # add some text for labels, title and axes ticks
    ax.set_ylabel(y_label, size = label_size)
    ax.set_title(' ')
    ax.set_xticks(ind + width)
    #tick_labels = dates.tolist()
    #ax.set_xticklabels(tick_labels)
    ax.grid(False)
    #plt.ylim([0,18])
    #make the border lines lighter
    [i.set_linewidth(0.1) for i in ax.spines.itervalues()]

    #remove the y axis
    #frame1 = plt.gca()
    #frame1.axes.get_yaxis().set_visible(False)

    #draw legend
    #ax.legend( (rects1[0], rects2[0]), ('Base Features', 'Best Features') , loc = 'best', fontsize = label_size, frameon=False)
    return fig


def plot_fitted_and_ref_vs_time(df, ref_column, xlim):
    plt.figure(facecolor='w', figsize = (10,3))
    a, b, axes, label_size = plot_params()
    df[ref_column].plot(marker = '.',linestyle = '-', label = 'Reference Data', xlim = xlim)
    #df.cv_lin_pred.plot(marker = '.',linestyle = '-', label = 'CV Predicted Data')
    df.cv_pred.plot(marker = '.',linestyle = '-', label = 'CV Predicted Data', xlim = xlim)

    #df.model_pred.plot(marker = '.',linestyle = '-', label = 'Linear Predicted Data')
    #axes.set_ylim([0,3])
    plt.legend(fontsize = label_size)
    plt.ylabel('Normalized Skier Numbers', size = label_size)
    plt.xlabel('Date', size = label_size)


def plot_lasso_and_ref_vs_time(df, ref_column, xlim):
    plt.figure(facecolor='w', figsize = (10,3))
    a, b, axes, label_size = plot_params()
    df[ref_column].plot(marker = '.',linestyle = '-', label = 'Reference Data', xlim = xlim)
    #df.cv_lin_pred.plot(marker = '.',linestyle = '-', label = 'CV Predicted Data')
    df.lasso_pred.plot(marker = '.',linestyle = '-', label = 'Lasso CV Predicted Data', xlim = xlim)

    #df.model_pred.plot(marker = '.',linestyle = '-', label = 'Linear Predicted Data')
    #axes.set_ylim([0,3])
    plt.legend(fontsize = label_size)
    plt.ylabel('Normalized Skier Numbers', size = label_size)
    plt.xlabel('Date', size = label_size)


def fitted_vs_ref_plot(df, ref_column):
    plt.figure(facecolor='w', figsize = (8,8))
    a, b, axes, label_size = plot_params()
    plt.plot(df[ref_column], df.lasso_pred, linestyle = '', marker = '.', alpha = 0.3)
    plt.xlabel('Skier Visits', size = label_size)
    plt.ylabel('Predicted Skier Visits', size = label_size)
    plt.plot([0, df.model_pred.max()], [0,df.model_pred.max()])
    #axes.set_ylim([-20,100])


def plot_error_vs_features(RMSE, ylim, xlim):
    plt.figure(facecolor='w', figsize = (10,5))
    a, b, axes, label_size = plot_params()
    x = range(1, len(RMSE)+1)
    plt.plot(x, RMSE, marker = '.', markersize = 20, label='RMSE')
    #axes.set_ylim(ylim)
    #axes.set_xlim(xlim)
    plt.xlabel('Number of Features', size = label_size)
    plt.ylabel('Error', size = label_size)
    #plt.grid(b=True, which='major', color='g', linestyle='-.')
    plt.legend(fontsize = label_size, loc = "best")


def plot_learning_curve(estimator, title, X, y, ylimit, days_tr, train_sizes):
    plt.figure(facecolor='w', figsize = (5,5), frameon = "True")
    a, b, axes, label_size = plot_params()
    plt.title(title, size = label_size)
    if ylimit is not None:
        axes.set_ylim(ylimit)
    plt.xlabel("Training Samples", size = label_size)
    plt.ylabel("Mean Squared Error", size = label_size)
    train_sizes, train_scores, valid_scores = learning_curve(estimator, X, y, cv = 5, train_sizes = train_sizes, scoring = 'mean_squared_error')
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = -np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    #plt.grid(b=True, which='major', color='#696969', linestyle=':')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
        alpha=0.1, color="r")
    plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std,
        alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training RMSE")
    plt.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Cross-validation RMSE")

    leg = plt.legend(loc="best", fontsize = label_size, frameon = 'True')
    leg.get_frame().set_facecolor('w')
    #fig.savefig('learning_curve.png', bbox_inches= 'tight')
    return plt


def plot_lambda(lambda_ridge, coefs, mean_score_lambda):
    #plot the coefficients
    plt.figure(facecolor='w', figsize = (10,5))
    a, b, axes, label_size = plot_params()
    #axes.set_ylim(-50, 50)
    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])

    ax.plot(lambda_ridge, coefs)
    ax.set_xscale('log')
    plt.xlabel('lambda', size = label_size)
    plt.ylabel('weights', size = label_size)
    plt.title('coefficients as a function of the regularization', size = label_size)
    plt.show()

    #plot the results
    plt.figure(facecolor='w', figsize = (10,5))
    plt.plot(lambda_ridge, mean_score_lambda)
    ax = plt.gca()
    ax.set_xscale('log')
    plt.xlabel('lambda', size= label_size)
    plt.ylabel('Custom Score', size = label_size)



if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))
