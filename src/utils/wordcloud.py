from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(text, title=None):
    """
    Generate and display a word cloud from the given text.

    Args:
        text (str): The text to generate the word cloud from.
        title (str): Optional title for the word cloud plot.
    """
    # Generate the word cloud
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='viridis',
        max_words=200
    ).generate(text)

    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off axes
    if title:
        plt.title(title, fontsize=16)
    plt.show()