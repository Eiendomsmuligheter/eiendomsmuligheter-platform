"""Add performance indexes

Revision ID: 002
Revises: 001
Create Date: 2025-02-03

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

def upgrade():
    # Properties indexes
    op.create_index('idx_properties_municipality_gnr_bnr', 'properties', 
                    ['municipality', 'gnr', 'bnr'])
    op.create_index('idx_properties_created_at', 'properties', ['created_at'])
    
    # Analyses indexes
    op.create_index('idx_analyses_property_id', 'analyses', ['property_id'])
    op.create_index('idx_analyses_status', 'analyses', ['status'])
    op.create_index('idx_analyses_created_at', 'analyses', ['created_at'])
    op.create_index('idx_analyses_completed_at', 'analyses', ['completed_at'])
    
    # Floor Plans indexes
    op.create_index('idx_floor_plans_property_id', 'floor_plans', 
                    ['property_id'])
    op.create_index('idx_floor_plans_floor_number', 'floor_plans', 
                    ['floor_number'])
    
    # Rooms indexes
    op.create_index('idx_rooms_floor_plan_id', 'rooms', ['floor_plan_id'])
    op.create_index('idx_rooms_room_type', 'rooms', ['room_type'])
    
    # Recommendations indexes
    op.create_index('idx_recommendations_analysis_id', 'recommendations', 
                    ['analysis_id'])
    op.create_index('idx_recommendations_complexity', 'recommendations', 
                    ['complexity'])
    
    # Documents indexes
    op.create_index('idx_documents_property_id', 'documents', ['property_id'])
    op.create_index('idx_documents_document_type', 'documents', 
                    ['document_type'])
    op.create_index('idx_documents_status', 'documents', ['status'])
    op.create_index('idx_documents_created_at', 'documents', ['created_at'])
    
    # Composite indexes for common queries
    op.create_index('idx_properties_analysis_status', 'analyses',
                    ['property_id', 'status', 'created_at'])
    op.create_index('idx_floor_plans_property_floor', 'floor_plans',
                    ['property_id', 'floor_number'])
    op.create_index('idx_documents_property_type', 'documents',
                    ['property_id', 'document_type', 'status'])

def downgrade():
    # Properties indexes
    op.drop_index('idx_properties_municipality_gnr_bnr')
    op.drop_index('idx_properties_created_at')
    
    # Analyses indexes
    op.drop_index('idx_analyses_property_id')
    op.drop_index('idx_analyses_status')
    op.drop_index('idx_analyses_created_at')
    op.drop_index('idx_analyses_completed_at')
    
    # Floor Plans indexes
    op.drop_index('idx_floor_plans_property_id')
    op.drop_index('idx_floor_plans_floor_number')
    
    # Rooms indexes
    op.drop_index('idx_rooms_floor_plan_id')
    op.drop_index('idx_rooms_room_type')
    
    # Recommendations indexes
    op.drop_index('idx_recommendations_analysis_id')
    op.drop_index('idx_recommendations_complexity')
    
    # Documents indexes
    op.drop_index('idx_documents_property_id')
    op.drop_index('idx_documents_document_type')
    op.drop_index('idx_documents_status')
    op.drop_index('idx_documents_created_at')
    
    # Composite indexes
    op.drop_index('idx_properties_analysis_status')
    op.drop_index('idx_floor_plans_property_floor')
    op.drop_index('idx_documents_property_type')