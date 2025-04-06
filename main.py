import gradio as gr
import numpy as np
from processing import process_image
from sudoku_solver import solve_sudoku
import copy


with gr.Blocks() as demo:
    with gr.Row():
        img_input = gr.Image(label="Upload Sudoku")
        grid_output = gr.Dataframe(label="Detected Sudoku Grid", 
                                   row_count=9, col_count=9, 
                                   datatype="number", 
                                   interactive=True)
    grid = gr.Dataframe(label="Editable Sudoku Grid", 
                            row_count=9, 
                            col_count=9, 
                            datatype="number", 
                            interactive=True)
    img_input.change(fn=process_image, inputs=img_input, outputs=grid_output)
    grid_output.change(fn=lambda x: x, inputs=grid_output, outputs=grid)
    btn = gr.Button("Solve")
    fixed_output = gr.Textbox(label="Solution")
    btn.click(fn=solve_sudoku, inputs=grid, outputs=fixed_output)

demo.launch()