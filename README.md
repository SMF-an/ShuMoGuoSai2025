### 全国大学生数学建模竞赛 LaTeX 论文模板  

本论文模板来自latexstudio的CUMCMThesis项目。https://github.com/latexstudio/CUMCMThesis.git

guide中介绍了一些关于latex的用法；template则提供了详细的论文模板，包括每部分应当写作的内容。（模板的内容来自b站的数学建模清风老师）

如果需要去掉封面并把论文标题保留在摘要上面，在加载类的使用如下语句：
```
    \documentclass[withoutpreface,bwprint]{cumcmthesis}
```
如果需要封面页，则是与原来一致：
```
    \documentclass[bwprint]{cumcmthesis}
```