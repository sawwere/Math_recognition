using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using static System.Net.Mime.MediaTypeNames;

namespace Viewer
{
    public partial class Form1 : Form
    {
        private string filepath = "";
        public Form1()
        {
            InitializeComponent();
        }

        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog openFileDialog = new OpenFileDialog())
            {
                openFileDialog1.Filter = "Jpg files (*.jpg)|*.jpg|All files (*.*)|*.*";
                if (openFileDialog1.ShowDialog() == DialogResult.OK)
                {

                    try
                    {
                        filepath = openFileDialog1.FileName;
                        pictureBox1.Image = new Bitmap(filepath);
                        pictureBox2.Image = new Bitmap(filepath);
                    }
                    catch
                    {
                        DialogResult result = MessageBox.Show("Could not open file",
                            "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Process p = Process.Start(new ProcessStartInfo
            {
                FileName = @" E:/Programs/Anaconda3/envs/MyPyTorch/python.exe",// GetPythonPath(),
                Arguments = @"t:/my_programs/Math_recognition/find.py 56 " + filepath + " REL",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                WindowStyle = ProcessWindowStyle.Hidden
            });
            string response = p.StandardOutput.ReadToEnd();
            Console.WriteLine(response);
            var response_parts = response.Split(' ');
            string base64_image = response_parts[0].Substring(2, response_parts[0].Length - 5);
            byte[] decodedImage = System.Convert.FromBase64String(base64_image);
            using (var ms = new MemoryStream(decodedImage))
            {
                pictureBox2.Image = new Bitmap(ms);
            }
        }
    }
}
