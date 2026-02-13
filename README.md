### 全国大学生数学建模竞赛 LaTeX 论文模板  

本论文模板来自 latexstudio 的 [CUMCMThesis](https://github.com/latexstudio/CUMCMThesis.git) 项目。

guide 中介绍了一些关于 latex 的用法；template 则提供了详细的论文模板，包括每部分应当写作的内容。（模板的内容来自b站的数学建模清风老师）

如果需要去掉封面并把论文标题保留在摘要上面，在加载类的使用如下语句：
```tex
\documentclass[withoutpreface,bwprint]{cumcmthesis}
```

如果需要封面页，则是与原来一致：
```tex
\documentclass[bwprint]{cumcmthesis}
```