import os
import shutil
import sqlite3
import tempfile
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file
)

from ai_engine import (
    evaluate_project,
    calculate_project_scores,
    generate_recommendations
)


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

app.secret_key = os.environ.get("SECRET_KEY", "projectiq-secret-key")


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATABASE_PATH = os.environ.get("DATABASE_PATH") or os.path.join(
    BASE_DIR,
    "projectiq.db"
)

DATASET_FOLDER = os.path.join(
    BASE_DIR,
    "datasets"
)

DATASET_PATH = os.path.join(
    DATASET_FOLDER,
    "project.csv"
)


# ============================================================
# DATABASE
# ============================================================

def prepare_database_file():

    # On Render the DB lives on a persistent disk; seed it once from
    # the bundled projectiq.db so existing records are not lost.
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

    bundled = os.path.join(BASE_DIR, "projectiq.db")

    if (
        not os.path.exists(DATABASE_PATH)
        and os.path.abspath(bundled) != os.path.abspath(DATABASE_PATH)
        and os.path.exists(bundled)
    ):
        shutil.copy(bundled, DATABASE_PATH)


def get_db_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_db():

    connection = get_db_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT,
            domain TEXT,
            users TEXT,
            technologies TEXT,
            uniqueness REAL,
            innovation REAL,
            feasibility REAL,
            resume REAL,
            overall REAL,
            recommendations TEXT,
            created_at TEXT
        )
    """)

    # Migrate columns if upgrading from earlier table schema
    cursor.execute("PRAGMA table_info(projects)")
    columns = [col[1] for col in cursor.fetchall()]

    if "resume" not in columns:
        if "resume_value" in columns:
            cursor.execute("ALTER TABLE projects ADD COLUMN resume REAL")
            cursor.execute("UPDATE projects SET resume = resume_value WHERE resume IS NULL")
        else:
            cursor.execute("ALTER TABLE projects ADD COLUMN resume REAL")

    if "recommendations" not in columns:
        cursor.execute("ALTER TABLE projects ADD COLUMN recommendations TEXT")

    if "created_at" not in columns:
        cursor.execute("ALTER TABLE projects ADD COLUMN created_at TEXT")

    connection.commit()
    connection.close()

# Auto-initialize database schema on load
prepare_database_file()
init_db()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# EVALUATE PROJECT
# ============================================================

@app.route("/evaluate", methods=["GET"])
def evaluate_page():
    return render_template("evaluate.html")


@app.route(
    "/evaluate",
    methods=["POST"]
)
def evaluate():

    try:

        title = request.form.get(
            "title",
            ""
        ).strip()

        description = request.form.get(
            "description",
            ""
        ).strip()

        domain = request.form.get(
            "domain",
            ""
        ).strip()

        users = request.form.get(
            "users",
            ""
        ).strip()

        technologies = request.form.get(
            "technologies",
            ""
        ).strip()


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not title:

            flash(
                "Please enter project title.",
                "error"
            )

            return redirect(
                url_for("index")
            )


        if not description:

            flash(
                "Please enter project description.",
                "error"
            )

            return redirect(
                url_for("index")
            )


        # ----------------------------------------------------
        # AI EVALUATION
        # ----------------------------------------------------

        uniqueness, similar_projects = evaluate_project(
            title=title,
            description=description,
            domain=domain,
            users=users,
            technologies=technologies
        )


        # ----------------------------------------------------
        # PROJECT SCORES
        # ----------------------------------------------------

        scores = calculate_project_scores(
            description,
            domain,
            users,
            technologies,
            uniqueness
        )


        # ----------------------------------------------------
        # RECOMMENDATIONS
        # ----------------------------------------------------

        recommendations = generate_recommendations(
            description,
            domain,
            uniqueness
        )


        # ----------------------------------------------------
        # SAVE TO DATABASE
        # ----------------------------------------------------

        connection = get_db_connection()

        cursor = connection.cursor()

        recommendations_text = "\n".join(
            recommendations
        )

        cursor.execute(
            """
            INSERT INTO projects (
                title,
                description,
                domain,
                users,
                technologies,
                uniqueness,
                innovation,
                feasibility,
                resume,
                overall,
                recommendations,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                description,
                domain,
                users,
                technologies,
                uniqueness,
                scores["innovation"],
                scores["feasibility"],
                scores["resume"],
                scores["overall"],
                recommendations_text,
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        )

        project_id = cursor.lastrowid

        connection.commit()
        connection.close()


        # ----------------------------------------------------
        # RESULT PAGE
        # ----------------------------------------------------

        return render_template(
            "result.html",
            project={
                "id": project_id,
                "title": title,
                "description": description,
                "domain": domain,
                "users": users,
                "technologies": technologies,
                "similarity": (
                    100 - uniqueness
                )
            },
            uniqueness=uniqueness,
            innovation=scores["innovation"],
            feasibility=scores["feasibility"],
            resume=scores["resume"],
            overall=scores["overall"],
            similar_projects=similar_projects,
            recommendations=recommendations
        )


    except Exception as e:

        print(
            "Evaluation error:",
            e
        )

        flash(
            "Something went wrong while evaluating the project.",
            "error"
        )

        return redirect(
            url_for("index")
        )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    connection = get_db_connection()

    projects = connection.execute(
        """
        SELECT *
        FROM projects
        ORDER BY id DESC
        """
    ).fetchall()

    connection.close()

    return render_template(
        "history.html",
        projects=projects
    )


# ============================================================
# PROJECT DETAILS
# ============================================================

@app.route(
    "/project/<int:project_id>"
)
def project_details(project_id):

    connection = get_db_connection()

    project = connection.execute(
        """
        SELECT *
        FROM projects
        WHERE id = ?
        """,
        (project_id,)
    ).fetchone()

    connection.close()


    if project is None:

        flash(
            "Project not found.",
            "error"
        )

        return redirect(
            url_for("history")
        )


    recommendations = []

    if project["recommendations"]:

        recommendations = [
            item.strip()
            for item in project[
                "recommendations"
            ].split("\n")
            if item.strip()
        ]

    # Dynamically find similar projects in the reference repository
    _, similar_projects = evaluate_project(
        title=project["title"] or "",
        description=project["description"] or "",
        domain=project["domain"] or "",
        users=project["users"] or "",
        technologies=project["technologies"] or "",
        include_db_projects=True,
        exclude_project_id=project["id"]
    )

    return render_template(
        "project_details.html",
        project=project,
        recommendations=recommendations,
        similar_projects=similar_projects
    )


# ============================================================
# DELETE PROJECT
# ============================================================

@app.route(
    "/delete/<int:project_id>",
    methods=["POST", "GET"]
)
def delete_project(project_id):

    connection = get_db_connection()

    connection.execute(
        """
        DELETE FROM projects
        WHERE id = ?
        """,
        (project_id,)
    )

    connection.commit()
    connection.close()

    flash(
        "Project deleted successfully.",
        "success"
    )

    return redirect(
        url_for("history")
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    connection = get_db_connection()

    projects = connection.execute(
        """
        SELECT *
        FROM projects
        ORDER BY id DESC
        """
    ).fetchall()

    connection.close()

    total_projects = len(projects)

    average_uniqueness = 0
    highest_project = None

    if total_projects > 0:
        average_uniqueness = round(
            sum(p["uniqueness"] for p in projects) / total_projects,
            1
        )
        highest_project = max(
            projects,
            key=lambda p: p["overall"]
        )

    return render_template(
        "dashboard.html",
        projects=projects,
        total_projects=total_projects,
        average_uniqueness=average_uniqueness,
        highest_project=highest_project
    )


# ============================================================
# COMPARE PROJECTS
# ============================================================

@app.route(
    "/compare",
    methods=["GET", "POST"]
)
def compare():

    connection = get_db_connection()

    projects = connection.execute(
        """
        SELECT *
        FROM projects
        ORDER BY id DESC
        """
    ).fetchall()

    connection.close()


    if request.method == "POST":

        # Support both checkbox list (project_ids) and
        # individual fields (project1, project2)
        project_ids = request.form.getlist("project_ids")

        if len(project_ids) == 2:
            project1_id = project_ids[0]
            project2_id = project_ids[1]
        else:
            project1_id = request.form.get("project1")
            project2_id = request.form.get("project2")

        if not project1_id or not project2_id:

            flash(
                "Please select exactly two projects to compare.",
                "error"
            )

            return redirect(
                url_for("history")
            )


        if project1_id == project2_id:

            flash(
                "Please select two different projects.",
                "error"
            )

            return redirect(
                url_for("history")
            )


        connection = get_db_connection()

        project1 = connection.execute(
            """
            SELECT *
            FROM projects
            WHERE id = ?
            """,
            (project1_id,)
        ).fetchone()

        project2 = connection.execute(
            """
            SELECT *
            FROM projects
            WHERE id = ?
            """,
            (project2_id,)
        ).fetchone()

        connection.close()


        return render_template(
            "compare.html",
            projects=projects,
            project1=project1,
            project2=project2
        )


    return render_template(
        "compare.html",
        projects=projects
    )


# ============================================================
# DATASET UPLOAD
# ============================================================

@app.route(
    "/upload-dataset",
    methods=["GET", "POST"]
)
def upload_dataset():

    if request.method == "POST":

        file = request.files.get(
            "file"
        )

        if not file:

            flash(
                "Please select a CSV file.",
                "error"
            )

            return redirect(
                url_for("upload_dataset")
            )


        filename = file.filename.lower()


        if not filename.endswith(".csv"):

            flash(
                "Only CSV files are allowed.",
                "error"
            )

            return redirect(
                url_for("upload_dataset")
            )


        # Create datasets folder if missing
        os.makedirs(
            DATASET_FOLDER,
            exist_ok=True
        )


        # Save as project.csv
        file.save(
            DATASET_PATH
        )


        flash(
            "Dataset uploaded successfully.",
            "success"
        )

        return redirect(
            url_for("upload_dataset")
        )


    return render_template(
        "upload_dataset.html"
    )


# ============================================================
# DOWNLOAD REPORT
# ============================================================

@app.route(
    "/download-report/<int:project_id>"
)
def download_report(project_id):

    """
    Generates a simple text report.

    If your existing project has a separate PDF generator,
    you can connect it here.
    """

    connection = get_db_connection()

    project = connection.execute(
        """
        SELECT *
        FROM projects
        WHERE id = ?
        """,
        (project_id,)
    ).fetchone()

    connection.close()


    if project is None:

        flash(
            "Project not found.",
            "error"
        )

        return redirect(
            url_for("history")
        )


    report_path = os.path.join(
        tempfile.gettempdir(),
        f"project_report_{project_id}.txt"
    )


    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "====================================\n"
        )

        file.write(
            "             PROJECT IQ\n"
        )

        file.write(
            "        PROJECT EVALUATION REPORT\n"
        )

        file.write(
            "====================================\n\n"
        )

        file.write(
            f"Project Title: {project['title']}\n"
        )

        file.write(
            f"Domain: {project['domain']}\n"
        )

        file.write(
            f"Users: {project['users']}\n"
        )

        file.write(
            f"Technologies: {project['technologies']}\n\n"
        )

        file.write(
            "------------------------------------\n"
        )

        file.write(
            "SCORES\n"
        )

        file.write(
            "------------------------------------\n"
        )

        file.write(
            f"Uniqueness: {project['uniqueness']}%\n"
        )

        file.write(
            f"Innovation: {project['innovation']}%\n"
        )

        file.write(
            f"Feasibility: {project['feasibility']}%\n"
        )

        file.write(
            f"Resume Value: {project['resume']}%\n"
        )

        file.write(
            f"Overall Score: {project['overall']}%\n\n"
        )

        file.write(
            "------------------------------------\n"
        )

        file.write(
            "DESCRIPTION\n"
        )

        file.write(
            "------------------------------------\n"
        )

        file.write(
            f"{project['description']}\n\n"
        )

        file.write(
            "------------------------------------\n"
        )

        file.write(
            "RECOMMENDATIONS\n"
        )

        file.write(
            "------------------------------------\n"
        )

        if project["recommendations"]:

            for recommendation in project[
                "recommendations"
            ].split("\n"):

                file.write(
                    f"- {recommendation}\n"
                )


    return send_file(
        report_path,
        as_attachment=True
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    # Create datasets folder automatically
    os.makedirs(
        DATASET_FOLDER,
        exist_ok=True
    )

    # Initialize database
    init_db()

    print(
        "\n======================================"
    )

    print(
        "        PROJECT IQ STARTED"
    )

    print(
        "======================================"
    )

    print(
        "Dataset:",
        DATASET_PATH
    )

    print(
        "Database:",
        DATABASE_PATH
    )

    print(
        "======================================\n"
    )

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "1") == "1"
    )