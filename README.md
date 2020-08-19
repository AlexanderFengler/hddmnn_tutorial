---


---

<h1 id="installation">Installation</h1>
<h2 id="non-local-install">Non-local install</h2>
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
<h2 id="local-install">Local install</h2>
<h3 id="windows">Windows</h3>
<p>Windows installation for hddm tutorial. We look to use the base image of a <code>miniconda</code> environment, then install everything into it.</p>
<p>Note: the installation process takes about thirty minutes or more. Please spare at least 8 GB on your machine and avoid intensive tasks during installation/downloading.</p>
<p>Note2: To copy in the command prompt and Conda bash, try Ctrl + C. If it doesn’t work, try Ctrl + Shift + C. To paste, try Ctrl + V, Ctrl + Shift + V, or right clicking in the terminal.</p>
<ol>
<li>Install  Git  for  Windows,  via  <a href="https://git-scm.com/download/win">https://git-scm.com/download/win</a></li>
<li>Install  Docker  for  Windows,  via  <a href="https://docs.docker.com/docker-for-windows/install/">https://docs.docker.com/docker-for-windows/install/</a>.  Please  pay  attention  to  the  requirements  to  make  sure  you  have  the  right  specs  for  Docker.</li>
<li>Download  the  Dockerfile  via  <a href="https://drive.google.com/drive/folders/1XU7PWNTp7aQRmLzQIhZJzItEW5ksR1TV?usp=sharing">https://drive.google.com/drive/folders/1XU7PWNTp7aQRmLzQIhZJzItEW5ksR1TV?usp=sharing</a></li>
<li>Run  Docker.</li>
<li>Navigate  to  the  downloaded  <code>hddm_tutorial</code>  folder,  hold  Shift  while  right-clicking  it  and  select  Copy  as  Path.</li>
<li>Open  Command  Prompt,  and  type  <code>docker --version</code>  to  make  sure  Docker  is  installed.</li>
<li><code>docker build &lt;paste-path-to-hddm_tutorial-here&gt;</code></li>
<li>Chill,  this  is  gonna  take  awhile.</li>
<li><code>docker image list</code>  to  see  if  the  image  is  installed  successfully.  Also  copy  the  Image  ID.</li>
<li>Rename  the  image  with  <code>docker image tag &lt;image-ID&gt; hddm_tutorial</code></li>
<li>Crseate  a  container  with  ports  <code>docker create -i -t -p 8888:8888 hddm_tutorial</code></li>
<li>Get  a  list  of  all  containers  <code>docker ps -a</code>,  and  copy  the  name  of  the  created  container  (it  should  be  something  funny  like  <code>positive_panini</code>)  and  whatnot.</li>
<li>Rename  the  container  <code>docker rename &lt;old-name&gt; hddm_container</code>.</li>
<li>Start  the  container  <code>docker start -i hddm_container</code></li>
<li>Once  you  are  in  the  Ubuntu  bash,  type  <code>conda deactivate</code></li>
<li><code>conda activate hddm4</code></li>
<li><code>jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root</code></li>
<li>Navigate  to  the  <code>tutorial</code>  Notebook,  and  begin!</li>
</ol>
<h3 id="ubuntu">Ubuntu</h3>
<blockquote>
<p>Written with <a href="https://stackedit.io/">StackEdit</a>.</p>
</blockquote>

