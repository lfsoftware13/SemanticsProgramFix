import subprocess

def run(tFile, dFile, rFile):
    build_cmd = 'python ' + tFile
    inFile = open(dFile,'r')
    outFile = open('C://Users//Administrator//Desktop//program_judgment//output.txt', 'w+')
    errorFile = open('C://Users//Administrator//Desktop//program_judgment//error.txt', 'w')
    try:
        subprocess.run(build_cmd, shell=True, stdin=inFile, stdout=outFile,stderr=errorFile, universal_newlines=True, timeout=4)
    except subprocess.TimeoutExpired as e:
        print('time out of limit')
    # p = subprocess.Popen(build_cmd,shell=True,stdin=inFile,stdout=outFile,stderr=errorFile, universal_newlines=True) #cwd设置工作目录
    # p.wait()#等待子进程执行结束

    outFile.seek(0,0)#移动文件指针到开头以便读取输出内容

    result = open(rFile, 'r')

    if outFile.read().strip('\n') == result.read().strip('\n'):
        errorFile.write('程序正确')
    else:
        errorFile.write('程序错误')

if __name__ == '__main__':
    #testFile = input("输入要进行测试的文件：")
    #dataFile = input("输入包含测试用例的文件：")
    #resultFile = input("输入包含正确结果的文件：")
    testFile = 'C://Users//Administrator//Desktop//program_judgment//test.py'
    dataFile = 'C://Users//Administrator//Desktop//program_judgment//data.txt'
    resultFile = 'C://Users//Administrator//Desktop//program_judgment//result.txt'
    run(testFile, dataFile, resultFile)
