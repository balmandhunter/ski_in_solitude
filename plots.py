import matplotlib.pyplot as plt
import seaborn as sns



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



if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))