def calculate_confidence(performance_score, practice_count):
	practice_bonus = min(max(practice_count, 0) * 2, 20)
	return round(min(max(performance_score + practice_bonus, 0), 100), 2)
