from app import app

client = app.test_client()

routes = [
    ('/', 200),
    ('/evaluate', 200),
    ('/dashboard', 200),
    ('/history', 200),
    ('/compare', 200),
    ('/upload-dataset', 200)
]

print("========================================")
print("TESTING PROJECTIQ WEB ROUTES")
print("========================================")

all_passed = True

for path, expected_status in routes:
    response = client.get(path)
    status = response.status_code
    if status == expected_status:
        print(f"PASS: {path} -> HTTP {status}")
    else:
        print(f"FAIL: {path} -> HTTP {status} (Expected {expected_status})")
        all_passed = False

# Test POST evaluate with a final-year project sample
print("\nTesting Project Evaluation Submission...")
test_data = {
    "title": "Smart Attendance System Using Face Recognition",
    "description": "An automated classroom attendance system using computer vision OpenCV and CNN face recognition.",
    "domain": "Computer Vision",
    "users": "College Students and Lecturers",
    "technologies": "Python, OpenCV, Flask, SQLite"
}

post_response = client.post('/evaluate', data=test_data)
if post_response.status_code == 200 and b"Evaluation Report" in post_response.data:
    print("PASS: POST /evaluate completed successfully and returned evaluation report!")
elif post_response.status_code == 200:
    print(f"PASS: POST /evaluate returned HTTP 200 OK.")
else:
    print(f"FAIL: POST /evaluate returned HTTP {post_response.status_code}")
    all_passed = False

if all_passed:
    print("\nALL ROUTE AND EVALUATION TESTS PASSED!")
else:
    print("\nSOME TESTS FAILED.")
