from dataclasses import dataclass


@dataclass(frozen=True)
class RoleSpec:
    """One job role used by the synthetic generator."""

    slug: str
    title: str
    family: str  # broad domain (engineering, data, product, design, ...)
    core_skills: tuple[str, ...]
    seniorities: tuple[str, ...]


ROLES: tuple[RoleSpec, ...] = (
    RoleSpec(
        "backend-dev",
        "Backend Engineer",
        "engineering",
        ("python", "fastapi", "postgres", "docker", "redis"),
        ("junior", "mid", "senior", "staff"),
    ),
    RoleSpec(
        "frontend-dev",
        "Frontend Engineer",
        "engineering",
        ("typescript", "react", "next.js", "css", "graphql"),
        ("junior", "mid", "senior", "staff"),
    ),
    RoleSpec(
        "fullstack-dev",
        "Full-Stack Engineer",
        "engineering",
        ("typescript", "react", "node.js", "postgres", "docker"),
        ("mid", "senior", "staff"),
    ),
    RoleSpec(
        "mobile-ios",
        "iOS Engineer",
        "engineering",
        ("swift", "swiftui", "xcode", "core data", "combine"),
        ("junior", "mid", "senior"),
    ),
    RoleSpec(
        "mobile-android",
        "Android Engineer",
        "engineering",
        ("kotlin", "jetpack compose", "android studio", "coroutines", "room"),
        ("junior", "mid", "senior"),
    ),
    RoleSpec(
        "devops",
        "DevOps Engineer",
        "engineering",
        ("kubernetes", "terraform", "aws", "ci/cd", "linux"),
        ("mid", "senior", "staff"),
    ),
    RoleSpec(
        "sre",
        "Site Reliability Engineer",
        "engineering",
        ("prometheus", "grafana", "kubernetes", "incident response", "slo"),
        ("mid", "senior", "staff"),
    ),
    RoleSpec(
        "ml-engineer",
        "Machine Learning Engineer",
        "engineering",
        ("python", "pytorch", "transformers", "mlops", "sql"),
        ("mid", "senior", "staff"),
    ),
    RoleSpec(
        "data-analyst",
        "Data Analyst",
        "data",
        ("sql", "tableau", "excel", "python", "a/b testing"),
        ("junior", "mid", "senior"),
    ),
    RoleSpec(
        "data-engineer",
        "Data Engineer",
        "data",
        ("python", "spark", "airflow", "snowflake", "dbt"),
        ("mid", "senior", "staff"),
    ),
    RoleSpec(
        "data-scientist",
        "Data Scientist",
        "data",
        ("python", "pandas", "scikit-learn", "statistics", "experimentation"),
        ("mid", "senior", "staff"),
    ),
    RoleSpec(
        "product-manager",
        "Product Manager",
        "product",
        ("roadmapping", "user research", "analytics", "stakeholder mgmt", "specs"),
        ("junior", "mid", "senior", "staff", "principal"),
    ),
    RoleSpec(
        "product-designer",
        "Product Designer",
        "design",
        ("figma", "user research", "prototyping", "design systems", "accessibility"),
        ("junior", "mid", "senior", "staff"),
    ),
    RoleSpec(
        "technical-writer",
        "Technical Writer",
        "documentation",
        ("docs-as-code", "markdown", "api docs", "information architecture", "git"),
        ("junior", "mid", "senior"),
    ),
    RoleSpec(
        "qa-engineer",
        "QA / Test Engineer",
        "engineering",
        ("playwright", "pytest", "ci/cd", "test design", "bug triage"),
        ("junior", "mid", "senior"),
    ),
)


def get_role(slug: str) -> RoleSpec:
    """Look up a RoleSpec by slug. Raises KeyError if unknown."""
    for role in ROLES:
        if role.slug == slug:
            return role
    raise KeyError(slug)
