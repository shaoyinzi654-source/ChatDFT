using System;
using System.ComponentModel;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

[assembly: AssemblyTitle("ChatDFT")]
[assembly: AssemblyDescription("ChatDFT 材料与分子计算工作台")]
[assembly: AssemblyCompany("ChatDFT")]
[assembly: AssemblyProduct("ChatDFT")]
[assembly: AssemblyVersion("5.9.0.0")]
[assembly: AssemblyFileVersion("5.9.0.0")]

internal static class ChatDFTLauncher
{
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr LoadLibraryEx(string path, IntPtr reserved, uint flags);

    [DllImport("kernel32.dll", CharSet = CharSet.Ansi, ExactSpelling = true, SetLastError = true)]
    private static extern IntPtr GetProcAddress(IntPtr library, string name);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int MessageBox(IntPtr window, string text, string title, uint flags);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int PythonMain(int count, [In, Out] IntPtr[] arguments);

    [STAThread]
    private static int Main(string[] args)
    {
        try
        {
            string root = AppDomain.CurrentDomain.BaseDirectory;
            string[] runtime = File.ReadAllLines(Path.Combine(root, "ChatDFT.runtime"));
            if (runtime.Length != 2 || !Path.IsPathRooted(runtime[0]) || !Path.IsPathRooted(runtime[1]))
                throw new InvalidDataException("启动配置无效。请重新运行 ChatDFT 安装脚本。");
            string script = Path.Combine(root, "runtime", "panel.py");
            if (!File.Exists(runtime[0]) || !File.Exists(runtime[1]) || !File.Exists(script))
                throw new FileNotFoundException("找不到 ChatDFT 运行环境或程序文件。请重新运行安装脚本。");

            const uint LoadFromDllDirectory = 0x00000100;
            const uint LoadFromDefaultDirectories = 0x00001000;
            IntPtr library = LoadLibraryEx(runtime[1], IntPtr.Zero, LoadFromDllDirectory | LoadFromDefaultDirectories);
            if (library == IntPtr.Zero)
                throw new Win32Exception(Marshal.GetLastWin32Error(), "无法加载 ChatDFT 的运行库。请检查安装环境。");
            IntPtr entry = GetProcAddress(library, "Py_Main");
            if (entry == IntPtr.Zero)
                throw new InvalidOperationException("运行库不兼容。请重新运行 ChatDFT 安装脚本。");

            // CPython locates pyvenv.cfg from argv[0], while Windows sees ChatDFT.exe.
            var command = new string[args.Length + 3];
            command[0] = runtime[0];
            command[1] = "-E";
            command[2] = script;
            Array.Copy(args, 0, command, 3, args.Length);
            var allocated = new IntPtr[command.Length];
            var argv = new IntPtr[command.Length + 1];
            try
            {
                for (int i = 0; i < command.Length; ++i)
                    argv[i] = allocated[i] = Marshal.StringToHGlobalUni(command[i]);
                var run = (PythonMain)Marshal.GetDelegateForFunctionPointer(entry, typeof(PythonMain));
                int exitCode = run(command.Length, argv);
                if (exitCode != 0)
                    MessageBox(IntPtr.Zero, "启动未完成，请核对 ChatDFT 的安装或启动参数。", "ChatDFT", 0x10);
                return exitCode;
            }
            finally
            {
                // Py_Main may change argv pointers. Free only the original allocations.
                foreach (IntPtr argument in allocated)
                    Marshal.FreeHGlobal(argument);
            }
        }
        catch (Exception error)
        {
            MessageBox(IntPtr.Zero, error.Message, "ChatDFT", 0x10);
            return 1;
        }
    }
}
