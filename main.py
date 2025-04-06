import gradio as gr
import numpy as np
from processing import process_image
from sudoku_solver import solve_and_return_grid

# Function to process the image and return grids
def process_for_both(img):
    result = process_image(img)
    return result  # Only return the processed grid for the editable grid


# Gradio UI
with gr.Blocks() as demo:
    with gr.Row():
        img_input = gr.Image(label="Upload Sudoku")

        # Editable grid
        grid_output = gr.Dataframe(
            label="Detected Sudoku Grid",
            row_count=9,
            col_count=9,
            datatype="number",
            interactive=True,  # Allow editing
        )



    # Process image once and display in grid_output
    img_input.change(fn=process_for_both, inputs=img_input, outputs=grid_output)
    print(grid_output)
    # Solve button to trigger solving
    btn = gr.Button("Solve")
    fixed_output = gr.Textbox(label="Solver Status")

    # When "Solve" is clicked → solve the user-edited grid
    btn.click(fn=solve_and_return_grid, inputs=grid_output, outputs=[grid_output, fixed_output])  # This saves and solves the edited grid

demo.launch()
