import os
import glob
import argparse
from pathlib import Path
import pandas as pd
import plotly.express as px
from ultralytics import YOLO

def benchmark_models(args):
    """
    Finds and benchmarks all .onnx models in a given folder for a specific task.
    """
    # Validate paths
    if not os.path.isdir(args.models_folder):
        print(f"❌ Error: Models folder not found at '{args.models_folder}'")
        return

    dataset_path = args.dataset
    if args.task == 'detection' and not os.path.isfile(dataset_path):
        print(f"❌ Error: Detection data.yaml not found at '{dataset_path}'")
        return
    if args.task == 'classification' and not os.path.isdir(dataset_path):
        print(f"❌ Error: Classification dataset folder not found at '{dataset_path}'")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📊 Plots and results will be saved to: '{args.output_dir}'")

    # Find all .onnx files
    onnx_files = glob.glob(os.path.join(args.models_folder, "*.onnx"))

    if not onnx_files:
        print(f"⚠️ Warning: No `.onnx` files found in `{args.models_folder}`.")
        return

    results_list = []
    total_models = len(onnx_files)
    print(f"\nFound {total_models} models to benchmark. Starting process...")

    for i, model_path in enumerate(onnx_files):
        model_name = os.path.basename(model_path)
        print(f"\n--- Benchmarking model {i+1}/{total_models}: {model_name} ---")

        try:
            model = YOLO(model_path)
            result_dict = {"model_name": model_name}

            if args.task == 'classification':
                metrics = model.val(data=dataset_path, imgsz=args.imgsz)
                result_dict['top1_accuracy'] = metrics.top1
                result_dict['top5_accuracy'] = metrics.top5
            
            elif args.task == 'detection':
                metrics = model.val(data=dataset_path, imgsz=args.imgsz)
                result_dict['mAP50-95'] = metrics.box.map
                result_dict['mAP50'] = metrics.box.map50
                result_dict['precision'] = metrics.box.mp
                result_dict['recall'] = metrics.box.mr
            
            # Common metrics
            result_dict['inference_speed_ms'] = metrics.speed['inference']
            results_list.append(result_dict)

        except Exception as e:
            print(f"❌ Failed to benchmark {model_name}. Error: {e}")

    if not results_list:
        print("Could not gather any results. Exiting.")
        return

    # --- Process and Display Results ---
    df = pd.DataFrame(results_list).round(4)
    df = df.sort_values(by=df.columns[1], ascending=False).reset_index(drop=True)

    print("\n\n✅ Benchmarking Complete!")
    print("\n--- Results Summary ---")
    print(df.to_string())

    # --- Generate and Save Plots ---
    print("\n--- Generating and Saving Plots ---")

    # Main performance plot
    if args.task == 'classification':
        y_metric = 'top1_accuracy'
        title = 'Top-1 Accuracy Comparison'
    else:
        y_metric = 'mAP50-95'
        title = 'mAP50-95 Comparison'

    fig_perf = px.bar(df, x='model_name', y=y_metric,
                      title=title,
                      labels={'model_name': 'Model', y_metric: y_metric},
                      color='model_name', text_auto=True)
    perf_path = os.path.join(args.output_dir, f"{args.task}_performance_comparison.png")
    fig_perf.write_image(perf_path, width=1200, height=700)
    print(f"Saved: {perf_path}")

    # Inference speed plot
    df_speed_sorted = df.sort_values('inference_speed_ms', ascending=True)
    fig_speed = px.bar(df_speed_sorted, x='model_name', y='inference_speed_ms',
                       title='Inference Speed (ms) per Image',
                       labels={'model_name': 'Model', 'inference_speed_ms': 'Time (ms)'},
                       color='model_name', text_auto=True)
    speed_path = os.path.join(args.output_dir, f"{args.task}_speed_comparison.png")
    fig_speed.write_image(speed_path, width=1200, height=700)
    print(f"Saved: {speed_path}")

def main():
    parser = argparse.ArgumentParser(description="CLI tool to benchmark YOLOv8 ONNX models and visualize performance.")
    
    parser.add_argument("--task", type=str, required=True, choices=['detection', 'classification'],
                        help="The task type for the models being benchmarked.")
    
    parser.add_argument("--models-folder", type=str, required=True,
                        help="Relative path to the folder containing your .onnx models.")
                        
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset. For detection, path to data.yaml. For classification, path to the root folder.")

    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory where results and plots will be saved. Defaults to 'benchmark_results'.")
                        
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for validation. Defaults to 640.")

    args = parser.parse_args()
    benchmark_models(args)

if __name__ == "__main__":
    main()
