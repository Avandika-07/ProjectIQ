from ai_engine import evaluate_project, calculate_project_scores, generate_recommendations

title = "AI Based Smart Waste Management System"
description = """
AI based smart waste management system
that predicts waste collection requirements
using machine learning and IoT sensors.
"""
domain = "IoT"
users = "Municipal Corporations and City Sanitation Workers"
technologies = "Python IoT Ultrasonic Sensors Flask Machine Learning"

uniqueness, similar_projects = evaluate_project(
    title=title,
    description=description,
    domain=domain,
    users=users,
    technologies=technologies
)

print(f"Project Title: {title}")
print(f"Domain: {domain}")
print(f"Uniqueness Score: {uniqueness}%\n")

print(f"Similar Projects Found ({len(similar_projects)}):")
for p in similar_projects[:5]:
    print(f" - {p['title']} [{p['domain']}]: {p['similarity']}% ({p['risk_level']})")
    if p.get("matched_terms"):
        print(f"   Matched: {', '.join(p['matched_terms'])}")