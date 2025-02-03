"""Initial database schema

Revision ID: 001
Revises: 
Create Date: 2025-02-03

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers, used by Alembic
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Properties table
    op.create_table(
        'properties',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('address', sa.String(), nullable=True),
        sa.Column('municipality', sa.String(), nullable=True),
        sa.Column('gnr', sa.Integer(), nullable=True),
        sa.Column('bnr', sa.Integer(), nullable=True),
        sa.Column('property_type', sa.String(), nullable=True),
        sa.Column('total_area', sa.Float(), nullable=True),
        sa.Column('building_area', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_properties_address'), 'properties', ['address'], unique=False)
    op.create_index(op.f('ix_properties_municipality'), 'properties', ['municipality'], unique=False)

    # Analyses table
    op.create_table(
        'analyses',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('property_id', sa.Integer(), nullable=True),
        sa.Column('analysis_type', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('result', JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['property_id'], ['properties.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Floor Plans table
    op.create_table(
        'floor_plans',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('property_id', sa.Integer(), nullable=True),
        sa.Column('floor_number', sa.Integer(), nullable=True),
        sa.Column('area', sa.Float(), nullable=True),
        sa.Column('ceiling_height', sa.Float(), nullable=True),
        sa.Column('file_path', sa.String(), nullable=True),
        sa.Column('model_url', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['property_id'], ['properties.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Rooms table
    op.create_table(
        'rooms',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('floor_plan_id', sa.Integer(), nullable=True),
        sa.Column('name', sa.String(), nullable=True),
        sa.Column('area', sa.Float(), nullable=True),
        sa.Column('width', sa.Float(), nullable=True),
        sa.Column('length', sa.Float(), nullable=True),
        sa.Column('height', sa.Float(), nullable=True),
        sa.Column('window_area', sa.Float(), nullable=True),
        sa.Column('door_count', sa.Integer(), nullable=True),
        sa.Column('room_type', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['floor_plan_id'], ['floor_plans.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Recommendations table
    op.create_table(
        'recommendations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('analysis_id', sa.Integer(), nullable=True),
        sa.Column('title', sa.String(), nullable=True),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('potential_value', sa.Float(), nullable=True),
        sa.Column('complexity', sa.String(), nullable=True),
        sa.Column('estimated_cost', sa.Float(), nullable=True),
        sa.Column('estimated_timeframe', sa.String(), nullable=True),
        sa.Column('requires_dispensation', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['analysis_id'], ['analyses.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Documents table
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('property_id', sa.Integer(), nullable=True),
        sa.Column('document_type', sa.String(), nullable=True),
        sa.Column('title', sa.String(), nullable=True),
        sa.Column('file_path', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['property_id'], ['properties.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(), nullable=True),
        sa.Column('hashed_password', sa.String(), nullable=True),
        sa.Column('full_name', sa.String(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('is_superuser', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    # User Analyses association table
    op.create_table(
        'user_analyses',
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('analysis_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['analysis_id'], ['analyses.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], )
    )

def downgrade():
    op.drop_table('user_analyses')
    op.drop_table('users')
    op.drop_table('documents')
    op.drop_table('recommendations')
    op.drop_table('rooms')
    op.drop_table('floor_plans')
    op.drop_table('analyses')
    op.drop_table('properties')