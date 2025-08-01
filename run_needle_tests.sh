# #!/bin/bash

# # Define the context lengths and document depth percentages
# CONTEXT_LENGTHS=(2000 10000 46000 73000 82000 91000 100000 109000 118000 128000)
# DOCUMENT_DEPTHS=(0 25 50 75 100)
# MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

# # Make sure the environment variables are set
# export EVALUATOR_MODEL_NAME="$MODEL_NAME"
# export OPENAI_API_BASE="http://localhost:8000/v1"
# export NIAH_MODEL_API_KEY="dummy-key"
# export NIAH_EVALUATOR_API_KEY="dummy-key"

# # Create a results directory with timestamp
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# RESULTS_DIR="needle_results_$TIMESTAMP"
# mkdir -p "$RESULTS_DIR"

# # Log file for the script
# LOG_FILE="$RESULTS_DIR/needle_test_$TIMESTAMP.log"

# echo "Starting needle-in-haystack tests at $(date)" | tee -a "$LOG_FILE"
# echo "Model: $MODEL_NAME" | tee -a "$LOG_FILE"
# echo "Results will be saved in: $RESULTS_DIR" | tee -a "$LOG_FILE"
# echo "---------------------------------------------" | tee -a "$LOG_FILE"

# # Function to run a single test and log results
# run_test() {
#     local context_length=$1
#     local document_depth=$2
    
#     echo "Running test: Context Length=$context_length, Document Depth=$document_depth%" | tee -a "$LOG_FILE"
    
#     # Create a unique output file for this test
#     output_file="$RESULTS_DIR/results_c${context_length}_d${document_depth}.txt"
    
#     # Run the test and capture output
#     needlehaystack.run_test \
#         --provider openai \
#         --model_name "$MODEL_NAME" \
#         --document_depth_percents "[$document_depth]" \
#         --context_lengths "[$context_length]" \
#         --save_contexts "false" \
#         2>&1 | tee "$output_file"
    
#     # Extract the score from the output file if possible
#     score=$(grep -o "Score: [0-9]*" "$output_file" | awk '{print $2}')
    
#     if [ -n "$score" ]; then
#         echo "Test completed with score: $score" | tee -a "$LOG_FILE"
#     else
#         echo "Test completed, but score could not be determined" | tee -a "$LOG_FILE"
#     fi
    
#     echo "---------------------------------------------" | tee -a "$LOG_FILE"
    
#     # Sleep to give the server a chance to reset
#     sleep 2
# }

# # Create a CSV summary file
# SUMMARY_FILE="$RESULTS_DIR/summary.csv"
# echo "Context Length,Document Depth %,Score" > "$SUMMARY_FILE"

# # Iterate through all combinations
# for context_length in "${CONTEXT_LENGTHS[@]}"; do
#     for depth in "${DOCUMENT_DEPTHS[@]}"; do
#         echo "Starting test: Context Length=$context_length, Document Depth=$depth%"
        
#         # Run the test
#         run_test "$context_length" "$depth"
        
#         # Extract score from the output file and append to summary
#         output_file="$RESULTS_DIR/results_c${context_length}_d${depth}.txt"
#         score=$(grep -o "Score: [0-9]*" "$output_file" | awk '{print $2}')
        
#         # Add to summary CSV
#         if [ -n "$score" ]; then
#             echo "$context_length,$depth,$score" >> "$SUMMARY_FILE"
#         else
#             echo "$context_length,$depth,N/A" >> "$SUMMARY_FILE"
#         fi
#     done
# done

# echo "All tests completed at $(date)" | tee -a "$LOG_FILE"
# echo "Summary saved to $SUMMARY_FILE" | tee -a "$LOG_FILE"
# echo "Results saved to $RESULTS_DIR" | tee -a "$LOG_FILE"

# # Generate a simple report
# echo "Generating report..."
# echo -e "\n\nSUMMARY OF RESULTS:\n" | tee -a "$LOG_FILE"
# echo "Context Length | 0% | 25% | 50% | 75% | 100%" | tee -a "$LOG_FILE"
# echo "-------------- | -- | --- | --- | --- | ----" | tee -a "$LOG_FILE"

# for context_length in "${CONTEXT_LENGTHS[@]}"; do
#     line="$context_length"
#     for depth in "${DOCUMENT_DEPTHS[@]}"; do
#         score=$(grep "^$context_length,$depth," "$SUMMARY_FILE" | cut -d ',' -f 3)
#         line="$line | $score"
#     done
#     echo "$line" | tee -a "$LOG_FILE"
# done

#!/bin/bash

# Define the context lengths and document depth percentages
CONTEXT_LENGTHS=(2000 6000 10000 15000 21000 25000 29000 33000 37000 41000)
CONTEXT_LENGTHS=(33000 37000 41000)
DOCUMENT_DEPTHS=(0 25 50 75 100)
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
REPEATS=2  # Run each test this many times

# Make sure the environment variables are set
export EVALUATOR_MODEL_NAME="$MODEL_NAME"
export OPENAI_API_BASE="http://localhost:8000/v1"
export NIAH_MODEL_API_KEY="dummy-key"
export NIAH_EVALUATOR_API_KEY="dummy-key"

# Create a results directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="needle_results_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

# Log file for the script
LOG_FILE="$RESULTS_DIR/needle_test_$TIMESTAMP.log"

echo "Starting needle-in-haystack tests at $(date)" | tee -a "$LOG_FILE"
echo "Model: $MODEL_NAME" | tee -a "$LOG_FILE"
echo "Each test will run $REPEATS times" | tee -a "$LOG_FILE"
echo "Results will be saved in: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "---------------------------------------------" | tee -a "$LOG_FILE"

# Function to run a single test and log results
run_test() {
    local context_length=$1
    local document_depth=$2
    local repeat_num=$3
    
    echo "Running test ($repeat_num/$REPEATS): Context Length=$context_length, Document Depth=$document_depth%" | tee -a "$LOG_FILE"
    
    # Create a unique output file for this test
    output_file="$RESULTS_DIR/results_c${context_length}_d${document_depth}_r${repeat_num}.txt"
    
    # Run the test and capture output
    needlehaystack.run_test \
        --provider openai \
        --model_name "$MODEL_NAME" \
        --document_depth_percents "[$document_depth]" \
        --context_lengths "[$context_length]" \
        --save_contexts "false" \
        --results_version "$repeat_num" \
        2>&1 | tee "$output_file"
    
    # Extract the score from the output file if possible
    score=$(grep -o "Score: [0-9]*" "$output_file" | awk '{print $2}')
    
    if [ -n "$score" ]; then
        echo "Test completed with score: $score" | tee -a "$LOG_FILE"
    else
        echo "Test completed, but score could not be determined" | tee -a "$LOG_FILE"
    fi
    
    echo "---------------------------------------------" | tee -a "$LOG_FILE"
    
    # Sleep to give the server a chance to reset
    sleep 2
}

# Create a CSV summary file
SUMMARY_FILE="$RESULTS_DIR/summary.csv"
echo "Context Length,Document Depth %,Run Number,Score" > "$SUMMARY_FILE"

# Iterate through all combinations
for context_length in "${CONTEXT_LENGTHS[@]}"; do
    for depth in "${DOCUMENT_DEPTHS[@]}"; do
        echo "Starting test set: Context Length=$context_length, Document Depth=$depth%"
        
        # Run each test the specified number of times
        for ((repeat=1; repeat<=$REPEATS; repeat++)); do
            # Run the test
            run_test "$context_length" "$depth" "$repeat"
            
            # Extract score from the output file and append to summary
            output_file="$RESULTS_DIR/results_c${context_length}_d${depth}_r${repeat}.txt"
            score=$(grep -o "Score: [0-9]*" "$output_file" | awk '{print $2}')
            
            # Add to summary CSV
            if [ -n "$score" ]; then
                echo "$context_length,$depth,$repeat,$score" >> "$SUMMARY_FILE"
            else
                echo "$context_length,$depth,$repeat,N/A" >> "$SUMMARY_FILE"
            fi
        done
    done
done

echo "All tests completed at $(date)" | tee -a "$LOG_FILE"
echo "Summary saved to $SUMMARY_FILE" | tee -a "$LOG_FILE"
echo "Results saved to $RESULTS_DIR" | tee -a "$LOG_FILE"

# Generate a simple report with averages
echo "Generating report with averages..."
echo -e "\n\nSUMMARY OF RESULTS (AVERAGES):\n" | tee -a "$LOG_FILE"
echo "Context Length | 0% | 25% | 50% | 75% | 100%" | tee -a "$LOG_FILE"
echo "-------------- | -- | --- | --- | --- | ----" | tee -a "$LOG_FILE"

# Average calculation function
calculate_avg() {
    local context=$1
    local depth=$2
    
    # Get all scores for this context/depth combination
    local scores=$(grep "^$context,$depth," "$SUMMARY_FILE" | cut -d ',' -f 4)
    
    # Calculate average if scores are available
    if [ -n "$scores" ]; then
        local sum=0
        local count=0
        for s in $scores; do
            if [[ "$s" =~ ^[0-9]+$ ]]; then
                sum=$((sum + s))
                count=$((count + 1))
            fi
        done
        
        if [ $count -gt 0 ]; then
            echo "$sum / $count" | bc -l | xargs printf "%.1f"
        else
            echo "N/A"
        fi
    else
        echo "N/A"
    fi
}

# Generate the averages report
for context_length in "${CONTEXT_LENGTHS[@]}"; do
    line="$context_length"
    for depth in "${DOCUMENT_DEPTHS[@]}"; do
        avg=$(calculate_avg "$context_length" "$depth")
        line="$line | $avg"
    done
    echo "$line" | tee -a "$LOG_FILE"
done

# Also create a detailed report showing individual runs
echo -e "\n\nDETAILED RESULTS (ALL RUNS):\n" | tee -a "$LOG_FILE"
echo "Context Length | Depth % | Run 1 | Run 2" | tee -a "$LOG_FILE"
echo "-------------- | ------- | ----- | -----" | tee -a "$LOG_FILE"

for context_length in "${CONTEXT_LENGTHS[@]}"; do
    for depth in "${DOCUMENT_DEPTHS[@]}"; do
        line="$context_length | $depth%"
        
        for ((repeat=1; repeat<=$REPEATS; repeat++)); do
            score=$(grep "^$context_length,$depth,$repeat," "$SUMMARY_FILE" | cut -d ',' -f 4)
            line="$line | ${score:-N/A}"
        done
        
        echo "$line" | tee -a "$LOG_FILE"
    done
done
