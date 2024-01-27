using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Viewer
{
    public partial class Form1 : Form
    {
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
                        var filename = openFileDialog1.FileName;
                        pictureBox1.Image = new Bitmap(filename);
                        pictureBox2.Image = new Bitmap(filename);
                    }
                    catch
                    {
                        DialogResult result = MessageBox.Show("Could not open file",
                            "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }
            }
        }
    }
}
