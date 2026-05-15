from training.dataset.roles import ROLES, RoleSpec, get_role


def test_role_catalog_has_at_least_fifteen_roles():
    assert len(ROLES) >= 15


def test_role_slugs_are_unique():
    slugs = [r.slug for r in ROLES]
    assert len(slugs) == len(set(slugs))


def test_role_seniorities_overlap_with_canonical_set():
    canonical = {"intern", "junior", "mid", "senior", "staff", "principal"}
    for role in ROLES:
        assert role.seniorities, f"{role.slug} has no seniorities"
        assert set(role.seniorities) <= canonical, role.slug


def test_role_families_cover_at_least_four_domains():
    families = {r.family for r in ROLES}
    assert len(families) >= 4


def test_get_role_by_slug_returns_the_role():
    role = get_role(ROLES[0].slug)
    assert isinstance(role, RoleSpec)
    assert role.slug == ROLES[0].slug
