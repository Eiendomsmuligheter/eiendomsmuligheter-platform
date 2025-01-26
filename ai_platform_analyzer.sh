#!/bin/bash

# Eiendomsmuligheter Platform - Advanced AI Context Generator
# Version: 2.0
# Date: 2025-01-26

# Set color codes for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PLATFORM_ROOT="$HOME/eiendomsmuligheter"
GITHUB_REPO="https://github.com/Eiendomsmuligheter/eiendomsmuligheter-platform.git"
AI_CONTEXT="$PLATFORM_ROOT/ai_context.md"
PROJECT_STATUS="$PLATFORM_ROOT/project_status.md"
DETAILED_ANALYSIS="$PLATFORM_ROOT/detailed_analysis.md"
COMPONENT_REPORT="$PLATFORM_ROOT/component_report.md"
CODE_METRICS="$PLATFORM_ROOT/code_metrics.md"
TECH_STACK="$PLATFORM_ROOT/tech_stack.md"
PROJECT_TIMELINE="$PLATFORM_ROOT/project_timeline.md"
DAILY_REPORT="$PLATFORM_ROOT/daily_report.md"

# Function to create timestamp
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Function to count lines of code for a specific file type
count_lines() {
    find "$PLATFORM_ROOT" -name "*.$1" -type f -exec wc -l {} \; | awk '{total += $1} END {print total}'
}

# Function to analyze code complexity
analyze_complexity() {
    local file=$1
    local ext="${file##*.}"
    
    case $ext in
        py)
            # Count function definitions and class declarations
            local functions=$(grep -c "^[[:space:]]*def" "$file")
            local classes=$(grep -c "^[[:space:]]*class" "$file")
            local imports=$(grep -c "^[[:space:]]*import\|^[[:space:]]*from" "$file")
            echo "Functions: $functions, Classes: $classes, Imports: $imports"
            ;;
        js)
            # Count function declarations and class definitions
            local functions=$(grep -c "function\|=>" "$file")
            local classes=$(grep -c "class" "$file")
            local imports=$(grep -c "import\|require" "$file")
            echo "Functions: $functions, Classes: $classes, Imports: $imports"
            ;;
        *)
            echo "Basic file"
            ;;
    esac
}

# Function to generate component dependency graph
generate_dependency_graph() {
    echo "# Component Dependency Analysis" > "$COMPONENT_REPORT"
    echo "Generated: $(timestamp)" >> "$COMPONENT_REPORT"
    echo "\`\`\`mermaid" >> "$COMPONENT_REPORT"
    echo "graph TD" >> "$COMPONENT_REPORT"
    
    # Analyze Python imports
    find "$PLATFORM_ROOT" -name "*.py" -type f | while read -r file; do
        filename=$(basename "$file" .py)
        grep "^import\|^from.*import" "$file" | while read -r line; do
            if [[ $line =~ ^from[[:space:]]+([^[:space:]]+)[[:space:]]+import ]]; then
                echo "    ${BASH_REMATCH[1]}-->$filename" >> "$COMPONENT_REPORT"
            fi
        done
    done
    
    echo "\`\`\`" >> "$COMPONENT_REPORT"
}

# Function to analyze code quality and patterns
analyze_code_quality() {
    local file=$1
    local ext="${file##*.}"
    
    echo "## Code Quality Analysis for $(basename "$file")" >> "$DETAILED_ANALYSIS"
    echo "### Analyzed on: $(timestamp)" >> "$DETAILED_ANALYSIS"
    
    # Check for common code patterns
    echo "#### Code Patterns:" >> "$DETAILED_ANALYSIS"
    case $ext in
        py)
            # Check for design patterns
            grep -l "class.*singleton" "$file" && echo "- Implements Singleton Pattern" >> "$DETAILED_ANALYSIS"
            grep -l "class.*factory" "$file" && echo "- Implements Factory Pattern" >> "$DETAILED_ANALYSIS"
            grep -l "class.*observer" "$file" && echo "- Implements Observer Pattern" >> "$DETAILED_ANALYSIS"
            
            # Check for best practices
            grep -l "def __init__" "$file" && echo "- Has proper class initialization" >> "$DETAILED_ANALYSIS"
            grep -l "try:" "$file" && echo "- Implements error handling" >> "$DETAILED_ANALYSIS"
            ;;
        js)
            # Check for modern JS patterns
            grep -l "async.*await" "$file" && echo "- Uses async/await pattern" >> "$DETAILED_ANALYSIS"
            grep -l "class.*extends" "$file" && echo "- Uses class inheritance" >> "$DETAILED_ANALYSIS"
            grep -l "const.*=.*=>" "$file" && echo "- Uses arrow functions" >> "$DETAILED_ANALYSIS"
            ;;
    esac
}

# Function to generate project metrics
generate_project_metrics() {
    echo "# Project Metrics Report" > "$CODE_METRICS"
    echo "Generated: $(timestamp)" >> "$CODE_METRICS"
    echo "" >> "$CODE_METRICS"
    
    # Count lines of code by language
    echo "## Lines of Code by Language" >> "$CODE_METRICS"
    echo "- Python: $(count_lines 'py') lines" >> "$CODE_METRICS"
    echo "- JavaScript: $(count_lines 'js') lines" >> "$CODE_METRICS"
    echo "- HTML: $(count_lines 'html') lines" >> "$CODE_METRICS"
    echo "- CSS: $(count_lines 'css') lines" >> "$CODE_METRICS"
    
    # Count files by type
    echo "" >> "$CODE_METRICS"
    echo "## File Count by Type" >> "$CODE_METRICS"
    echo "- Python Files: $(find "$PLATFORM_ROOT" -name "*.py" | wc -l)" >> "$CODE_METRICS"
    echo "- JavaScript Files: $(find "$PLATFORM_ROOT" -name "*.js" | wc -l)" >> "$CODE_METRICS"
    echo "- HTML Files: $(find "$PLATFORM_ROOT" -name "*.html" | wc -l)" >> "$CODE_METRICS"
    echo "- CSS Files: $(find "$PLATFORM_ROOT" -name "*.css" | wc -l)" >> "$CODE_METRICS"
    echo "- Documentation Files: $(find "$PLATFORM_ROOT" -name "*.md" | wc -l)" >> "$CODE_METRICS"
}

# Function to analyze technology stack
analyze_tech_stack() {
    echo "# Technology Stack Analysis" > "$TECH_STACK"
    echo "Generated: $(timestamp)" >> "$TECH_STACK"
    echo "" >> "$TECH_STACK"
    
    # Analyze Python dependencies
    if [ -f "$PLATFORM_ROOT/requirements.txt" ]; then
        echo "## Python Dependencies" >> "$TECH_STACK"
        cat "$PLATFORM_ROOT/requirements.txt" >> "$TECH_STACK"
        echo "" >> "$TECH_STACK"
    fi
    
    # Analyze JavaScript dependencies
    if [ -f "$PLATFORM_ROOT/package.json" ]; then
        echo "## JavaScript Dependencies" >> "$TECH_STACK"
        jq '.dependencies' "$PLATFORM_ROOT/package.json" >> "$TECH_STACK"
        echo "" >> "$TECH_STACK"
    fi
    
    # Analyze Docker configuration
    if [ -f "$PLATFORM_ROOT/Dockerfile" ]; then
        echo "## Docker Configuration" >> "$TECH_STACK"
        cat "$PLATFORM_ROOT/Dockerfile" >> "$TECH_STACK"
        echo "" >> "$TECH_STACK"
    fi
}

# Function to generate AI context
generate_ai_context() {
    echo "# Eiendomsmuligheter Platform - AI Context" > "$AI_CONTEXT"
    echo "Last Updated: $(timestamp)" >> "$AI_CONTEXT"
    echo "" >> "$AI_CONTEXT"
    
    # Include project overview
    echo "## Project Overview" >> "$AI_CONTEXT"
    echo "- Repository: $GITHUB_REPO" >> "$AI_CONTEXT"
    echo "- Root Directory: $PLATFORM_ROOT" >> "$AI_CONTEXT"
    echo "" >> "$AI_CONTEXT"
    
    # Include metrics summary
    echo "## Project Metrics" >> "$AI_CONTEXT"
    echo "\`\`\`" >> "$AI_CONTEXT"
    cat "$CODE_METRICS" >> "$AI_CONTEXT"
    echo "\`\`\`" >> "$AI_CONTEXT"
    echo "" >> "$AI_CONTEXT"
    
    # Include component analysis
    echo "## Component Analysis" >> "$AI_CONTEXT"
    cat "$COMPONENT_REPORT" >> "$AI_CONTEXT"
    echo "" >> "$AI_CONTEXT"
    
    # Include tech stack
    echo "## Technology Stack" >> "$AI_CONTEXT"
    cat "$TECH_STACK" >> "$AI_CONTEXT"
    echo "" >> "$AI_CONTEXT"
    
    # Include detailed analysis
    echo "## Detailed Analysis" >> "$AI_CONTEXT"
    cat "$DETAILED_ANALYSIS" >> "$AI_CONTEXT"
    echo "" >> "$AI_CONTEXT"
    
    # Add AI instructions
    echo "## AI Assistant Instructions" >> "$AI_CONTEXT"
    echo "1. Always check the entire context before starting work" >> "$AI_CONTEXT"
    echo "2. Reference specific files and line numbers when discussing code" >> "$AI_CONTEXT"
    echo "3. Update project_log.md with all changes" >> "$AI_CONTEXT"
    echo "4. Generate daily progress report at start of each session" >> "$AI_CONTEXT"
    echo "5. Keep track of project timeline and deadlines" >> "$AI_CONTEXT"
    echo "" >> "$AI_CONTEXT"
    
    # Add project status summary
    echo "## Project Status Summary" >> "$AI_CONTEXT"
    echo "- Frontend Status: $(calculate_frontend_progress)%" >> "$AI_CONTEXT"
    echo "- Backend Status: $(calculate_backend_progress)%" >> "$AI_CONTEXT"
    echo "- Test Coverage: $(calculate_test_coverage)%" >> "$AI_CONTEXT"
    echo "- Documentation Status: $(calculate_documentation_status)%" >> "$AI_CONTEXT"
}

# Function to calculate progress percentages
calculate_frontend_progress() {
    # Add your frontend progress calculation logic here
    echo "75"
}

calculate_backend_progress() {
    # Add your backend progress calculation logic here
    echo "80"
}

calculate_test_coverage() {
    # Add your test coverage calculation logic here
    echo "60"
}

calculate_documentation_status() {
    # Add your documentation status calculation logic here
    echo "70"
}

# Function to generate daily report
generate_daily_report() {
    echo "# Daily Project Report" > "$DAILY_REPORT"
    echo "Generated: $(timestamp)" >> "$DAILY_REPORT"
    echo "" >> "$DAILY_REPORT"
    
    # Include project status
    echo "## Current Project Status" >> "$DAILY_REPORT"
    echo "- Overall Progress: $(calculate_overall_progress)%" >> "$DAILY_REPORT"
    echo "- Frontend Progress: $(calculate_frontend_progress)%" >> "$DAILY_REPORT"
    echo "- Backend Progress: $(calculate_backend_progress)%" >> "$DAILY_REPORT"
    echo "- Test Coverage: $(calculate_test_coverage)%" >> "$DAILY_REPORT"
    echo "" >> "$DAILY_REPORT"
    
    # Include recent changes
    echo "## Recent Changes" >> "$DAILY_REPORT"
    git -C "$PLATFORM_ROOT" log --since="1 day ago" --pretty=format:"- %s" >> "$DAILY_REPORT"
    echo "" >> "$DAILY_REPORT"
    
    # Include pending tasks
    echo "## Pending Tasks" >> "$DAILY_REPORT"
    if [ -f "$PLATFORM_ROOT/TODO.md" ]; then
        cat "$PLATFORM_ROOT/TODO.md" >> "$DAILY_REPORT"
    fi
    echo "" >> "$DAILY_REPORT"
    
    # Include system health
    echo "## System Health" >> "$DAILY_REPORT"
    echo "- Disk Usage: $(df -h / | awk 'NR==2 {print $5}')" >> "$DAILY_REPORT"
    echo "- Memory Usage: $(free -h | awk 'NR==2 {print $3"/"$2}')" >> "$DAILY_REPORT"
    echo "- CPU Load: $(uptime | awk -F'load average:' '{print $2}')" >> "$DAILY_REPORT"
}

# Main execution
main() {
    echo -e "${GREEN}Starting Eiendomsmuligheter Platform Analysis...${NC}"
    
    # Create necessary directories
    mkdir -p "$PLATFORM_ROOT"
    
    # Generate all reports
    echo -e "${BLUE}Generating project metrics...${NC}"
    generate_project_metrics
    
    echo -e "${BLUE}Analyzing code quality...${NC}"
    find "$PLATFORM_ROOT" -type f \( -name "*.py" -o -name "*.js" \) -exec bash -c 'analyze_code_quality "$0"' {} \;
    
    echo -e "${BLUE}Generating dependency graph...${NC}"
    generate_dependency_graph
    
    echo -e "${BLUE}Analyzing technology stack...${NC}"
    analyze_tech_stack
    
    echo -e "${BLUE}Generating AI context...${NC}"
    generate_ai_context
    
    echo -e "${BLUE}Generating daily report...${NC}"
    generate_daily_report
    
    # Set up automatic backup
    if ! crontab -l | grep -q "$PLATFORM_ROOT"; then
        (crontab -l 2>/dev/null; echo "*/5 * * * * cd $PLATFORM_ROOT && git add . && git commit -m 'Auto-backup $(date)' && git push") | crontab -
    fi
    
    echo -e "${GREEN}Analysis complete! AI context and reports have been generated.${NC}"
    echo -e "${YELLOW}AI Assistant should now have full context of the project.${NC}"
}

# Execute main function
main

# Set up file watchers for real-time updates
inotifywait -m -r -e modify,create,delete "$PLATFORM_ROOT" |
while read -r directory events filename; do
    generate_ai_context
    generate_daily_report
done