"""ratings history (predicted-vs-actual, docs/07 D)

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE ratings (
            user_id     text,
            contest_id  int,
            new_rating  int,
            update_time bigint,
            PRIMARY KEY (user_id, contest_id)
        );
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS ratings;")
