---


---

<h1 id="installation">Installation</h1>
<p>We use Google Colab to clone the tutorial Repository onto your Google Drive, and then from there open the tutorial Jupyter Notebook via Colab.</p>
<ol>
<li>Make a dummy Colab Textbook: visit <a href="http://colab.research.google.com">colab.research.google.com</a>, then create a Colab Notebook.</li>
<li>In the first cell, paste in<br> <code>from google.colab import drive</code><br>
<code>drive.mount('/content/gdrive')</code>. Run the cell.</li>
<li>Click on the output URL, give the correct authentication code.</li>
<li>Make a new cell (you can also use the old cell, it doesn’t really matter), and paste in<br><code>%cd gdrive/'My Drive'/</code>.  Check that you are in the correct directory by running the command <code>!ls</code> in a cell.</li>
<li>Clone the repo by running a cell with <code>git clone https://github.com/AlexanderFengler/hddmnn_tutorial</code></li>
<li>Open your Google Drive, navigate to the <code>hddmnn_tutorial</code> folder, and open <code>ddm_variants_exploration.ipynb</code>, then select Google Colab.</li>
</ol>
<blockquote>
<p>Written with <a href="https://stackedit.io/">StackEdit</a>.</p>
</blockquote>

