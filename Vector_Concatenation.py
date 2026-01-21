import pandas as pd
df = pd.read_csv("C:\TOTbaseline\datascience\split_song_lyrics_with_BERT2_emotions_en_100k_2.csv")
result2 = df.groupby("title")[["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]].sum()
print(result2.head())
result2.to_csv("C:\TOTbaseline\datascience\song_emotions_vector_2.csv")