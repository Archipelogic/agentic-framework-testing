#!/bin/bash
# ============================================================================
# Agentic AI Framework Testing Harness - Unified Runner
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Display header
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     🚀 Agentic AI Framework Testing Harness 🚀           ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to show help
show_help() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

COMMANDS:
  setup       Install dependencies and prepare environment
  test        Run framework evaluation tests
  clean       Clean benchmark results and cache files
  report      Open the latest report in browser
  help        Show this help message

OPTIONS (for 'test' command):
  --mock      Run with mock adapters (no API keys required) [default]
  --live      Run with real API calls (requires API keys)
  --quick     Quick mode: test top 3 frameworks on 2 use cases
  --samples N Number of test samples per use case (default: 20)
  --parallel  Run tests in parallel (live mode only)
  --no-open   Don't auto-open the HTML report

EXAMPLES:
  $0 setup                    # Set up environment
  $0 test                     # Run mock evaluation
  $0 test --live              # Run with real APIs
  $0 test --mock --samples 50 # Run mock with 50 samples
  $0 report                   # Open latest report
  $0 clean                    # Clean all results

EOF
}

# Function to check Python
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+${NC}"
        exit 1
    fi
    
    python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
    required_version="3.8"
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
        echo -e "${GREEN}✅ Python $python_version detected${NC}"
    else
        echo -e "${RED}❌ Python $python_version found. Python 3.8+ required${NC}"
        exit 1
    fi
}

# Function to setup environment
setup_environment() {
    echo -e "${BLUE}📦 Setting up environment...${NC}\n"
    
    check_python
    
    # Create virtual environment if needed
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv venv
        echo -e "${GREEN}✅ Virtual environment created${NC}\n"
    else
        echo -e "${GREEN}✅ Virtual environment exists${NC}\n"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    echo -e "${GREEN}✅ Virtual environment activated${NC}\n"
    
    # Upgrade pip
    echo -e "${YELLOW}Upgrading pip...${NC}"
    pip install --upgrade pip --quiet
    echo -e "${GREEN}✅ Pip upgraded${NC}\n"
    
    # Install dependencies
    echo -e "${YELLOW}Installing dependencies...${NC}"
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt --quiet
        echo -e "${GREEN}✅ Dependencies installed${NC}\n"
    fi
    
    # Install package in development mode
    if [ -f "setup.py" ]; then
        echo -e "${YELLOW}Installing package in development mode...${NC}"
        pip install -e . --quiet
        echo -e "${GREEN}✅ Package installed${NC}\n"
    fi
    
    # Create necessary directories
    echo -e "${YELLOW}Creating project directories...${NC}"
    mkdir -p benchmark_results
    mkdir -p data/{movie_recommendation,github_triage,recipe_generation,research_summary,email_automation}
    echo -e "${GREEN}✅ Directories created${NC}\n"
    
    # Create .env file if needed
    if [ ! -f ".env" ] && [ -f ".env.example" ]; then
        echo -e "${YELLOW}Creating .env file from template...${NC}"
        cp .env.example .env
        echo -e "${YELLOW}⚠️  Please edit .env file to add your API keys${NC}\n"
    fi
    
    # Generate test data if needed
    echo -e "${YELLOW}Checking test data...${NC}"
    python3 -c "
from src.utils.data_generator import ensure_all_data_exists
success, failed = ensure_all_data_exists(verbose=True)
if not success:
    print(f'❌ Failed to generate data for: {\", \".join(failed)}')
    exit(1)
"
    
    echo -e "\n${GREEN}✅ Setup complete!${NC}"
}

# Function to run tests
run_tests() {
    # Parse test options
    MODE="mock"
    QUICK=""
    SAMPLES=""
    PARALLEL=""
    NO_OPEN=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mock)
                MODE="mock"
                shift
                ;;
            --live)
                MODE="live"
                shift
                ;;
            --quick)
                QUICK="--quick"
                shift
                ;;
            --samples)
                SAMPLES="--samples $2"
                shift 2
                ;;
            --parallel)
                PARALLEL="--parallel"
                shift
                ;;
            --no-open)
                NO_OPEN="--no-open"
                shift
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Check virtual environment and activate
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}⚠️  Virtual environment not found. Running setup first...${NC}"
        setup_environment
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Check for API keys if running in live mode
    if [ "$MODE" = "live" ]; then
        echo -e "${BLUE}🔍 Checking API keys...${NC}"
        has_keys=false
        
        # Check for any API keys
        if [ ! -z "$OPENAI_API_KEY" ] || [ ! -z "$ANTHROPIC_API_KEY" ] || [ ! -z "$AWS_ACCESS_KEY_ID" ] || [ -f ".env" ]; then
            has_keys=true
        fi
        
        # Check for AWS Bedrock configuration
        if [ ! -z "$AWS_ACCESS_KEY_ID" ] && [ ! -z "$AWS_SECRET_ACCESS_KEY" ]; then
            echo -e "${GREEN}✅ AWS Bedrock credentials detected${NC}"
            export AWS_BEDROCK_ENABLED=true
            if [ -z "$AWS_REGION" ]; then
                export AWS_REGION="us-east-1"
                echo -e "${YELLOW}   Using default AWS region: us-east-1${NC}"
            fi
        fi
        
        if [ "$has_keys" = false ]; then
            echo -e "${YELLOW}⚠️  No API keys found. Please set environment variables or create .env file${NC}"
            echo -e "${YELLOW}   Falling back to mock mode...${NC}\n"
            MODE="mock"
        fi
    fi
    
    # Build command
    CMD="python3 run_evaluation.py --$MODE $QUICK $SAMPLES $PARALLEL $NO_OPEN"
    
    echo -e "${BLUE}🚀 Running framework evaluation...${NC}"
    echo -e "${YELLOW}Mode: $MODE${NC}"
    
    # Run the evaluation
    eval $CMD
}

# Function to clean results
clean_results() {
    echo -e "${YELLOW}🧹 Cleaning benchmark results...${NC}"
    
    if [ -d "benchmark_results" ] && [ "$(ls -A benchmark_results)" ]; then
        read -p "Are you sure you want to delete all results? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf benchmark_results/*
            echo -e "${GREEN}✅ Results cleaned${NC}"
        else
            echo -e "${YELLOW}Cancelled${NC}"
        fi
    else
        echo -e "${GREEN}✅ No results to clean${NC}"
    fi
    
    # Clean Python cache
    echo -e "${YELLOW}Cleaning Python cache...${NC}"
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    echo -e "${GREEN}✅ Cache cleaned${NC}"
}

# Function to open latest report
open_report() {
    echo -e "${BLUE}📊 Opening latest report...${NC}"
    
    # Find the most recent report
    if [ -d "benchmark_results" ]; then
        latest_dir=$(ls -td benchmark_results/*/ 2>/dev/null | head -1)
        if [ ! -z "$latest_dir" ]; then
            report_file="${latest_dir}report.html"
            if [ -f "$report_file" ]; then
                echo -e "${GREEN}Found report: $report_file${NC}"
                if [[ "$OSTYPE" == "darwin"* ]]; then
                    open "$report_file"
                elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
                    xdg-open "$report_file"
                elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
                    start "$report_file"
                else
                    echo -e "${YELLOW}Please open manually: $report_file${NC}"
                fi
                echo -e "${GREEN}✅ Report opened${NC}"
            else
                echo -e "${YELLOW}No report.html found in latest results${NC}"
            fi
        else
            echo -e "${YELLOW}No benchmark results found${NC}"
        fi
    else
        echo -e "${YELLOW}No benchmark results directory found${NC}"
    fi
}

# Main execution
case "${1:-help}" in
    setup)
        setup_environment
        ;;
    test)
        shift
        run_tests "$@"
        ;;
    clean)
        clean_results
        ;;
    report)
        open_report
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac
