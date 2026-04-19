import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


sns.set_theme(style="darkgrid", context="talk")


def plot_rewards(total_rewards):

	df = pd.DataFrame({
		"episode": list(range(len(total_rewards))),
		"reward": total_rewards
	})

	df["smoothed"] = df["reward"].rolling(window=200).mean()

	plt.figure(figsize=(12, 6))
	sns.lineplot(data=df, x="episode", y="reward", alpha=0.2, label="Raw")
	sns.lineplot(data=df, x="episode", y="smoothed", linewidth=3, label="Smoothed")
	plt.title("Training Reward")
	plt.show()



def plot_scores(total_scores):

	plt.figure(figsize=(10, 6))
	sns.histplot(total_scores, bins=40, kde=True)
	plt.title("Score Distribution")
	plt.xlabel("score")
	plt.show()



def plot_scores_over_time(total_scores):

	chunk_size = 1000

	chunks = [
		total_scores[i:i+chunk_size]
		for i in range(0, len(total_scores), chunk_size)
	]

	df_chunks = pd.DataFrame({
		"score": [s for chunk in chunks for s in chunk],
		"chunk": [i for i, chunk in enumerate(chunks) for _ in chunk]
	})

	plt.figure(figsize=(12, 6))
	sns.boxplot(data=df_chunks, x="chunk", y="score")
	plt.title("Score Distribution Over Time")
	plt.show()



def plot_action_distribution(action_counts):

	actions = list(range(len(action_counts)))

	df_actions = pd.DataFrame({
		"action": actions,
		"count": action_counts
	})

	plt.figure(figsize=(10, 6))
	sns.barplot(data=df_actions, x="action", y="count")
	plt.title("Final Action Distribution")
	plt.show()



def plot_reward_vs_score(total_rewards, total_scores):

	df = pd.DataFrame({
		"reward": total_rewards,
		"score": total_scores
	})

	plt.figure(figsize=(8, 6))
	sns.scatterplot(data=df, x="reward", y="score", alpha=0.5)
	sns.regplot(data=df, x="reward", y="score", scatter=False, color="red")
	plt.title("Reward vs Score")
	plt.show()



def plot_rolling_score(total_scores):

	df = pd.DataFrame({
		"episode": list(range(len(total_scores))),
		"score": total_scores
	})

	df["rolling"] = df["score"].rolling(200).mean()

	plt.figure(figsize=(12, 6))
	sns.lineplot(data=df, x="episode", y="rolling", linewidth=3)
	plt.title("Rolling Average Score")
	plt.show()



def plot_strategy_performance(action_history, total_scores):

	df = pd.DataFrame({
		"action": action_history,
		"score": total_scores
	})

	plt.figure(figsize=(10, 6))
	sns.boxplot(data=df, x="action", y="score")
	plt.title("Score per Strategy")
	plt.show()