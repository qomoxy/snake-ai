import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, means_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number of Games : ")
    plt.ylabel("Score : ")
    plt.plot(scores)
    plt.plot(means_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(means_scores)-1, means_scores[-1], str(means_scores[-1]))

