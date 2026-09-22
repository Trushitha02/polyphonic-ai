def predict_level(score):
	if score >= 90:
		return "Advanced"
	if score >= 70:
		return "Intermediate"
	return "Beginner"
