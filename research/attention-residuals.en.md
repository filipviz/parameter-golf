---
title: "A Memoir of Attention Residuals - Scientific Spaces|Scientific Spaces"
source: "https://kexue.fm/archives/11664"
author:
published:
created: 2026-03-19
description: "This article introduces our latest work, Attention Residuals (AttnRes). As the name suggests, it improves residual connections using the idea of attention. Many readers have probably heard about the Pre Norm/Post Norm debate..."
tags:
  - "clippings"
---
19 Mar

## A Memoir of Attention Residuals

By Jianlin Su | 2026-03-19 | 5246 readers |

This article introduces one of our latest works, [Attention Residuals (AttnRes)](https://papers.cool/arxiv/2603.15031). As the name suggests, it uses the idea of Attention to improve Residuals.

Many readers have probably heard about the Pre Norm/Post Norm debate, but in the end that is still just an “internal struggle” within Residuals themselves, and the many later changes to Normalization are much the same. A more interesting variation is [HC](https://papers.cool/arxiv/2409.19606), which began exploring the route of widening the residual stream, but perhaps because its gains were unstable, it did not attract much attention. The rest of the story is probably familiar: at the end of last year, DeepSeek’s [mHC](https://papers.cool/arxiv/2512.24880) improved HC and validated its effectiveness at larger scale.

Instead of further widening the residual stream, we chose another radical route: directly applying Attention across layers to replace Residuals. Of course, making the whole pipeline work involved many details and a lot of effort, so here I will simply look back on the path that led to it.

[![AttnRes schematic](https://kexue.fm/usr/uploads/2026/03/1124653441.png)](https://kexue.fm/usr/uploads/2026/03/1124653441.png "Click to view full image")

AttnRes schematic

## Inter-Layer Attention

As usual, let us begin with [Residuals](https://papers.cool/arxiv/1512.03385), which should be familiar to everyone. Its form is  
$$
\begin{equation}\boldsymbol{x}_t = \boldsymbol{x}_{t-1} + \boldsymbol{f}_t(\boldsymbol{x}_{t-1})\end{equation}
$$
  
Here let us rewrite it in another way, which reveals something deeper. First define $\boldsymbol{y}_t = \boldsymbol{f}_t(\boldsymbol{x}_{t-1})$, then $\boldsymbol{x}_t = \boldsymbol{x}_{t-1} + \boldsymbol{y}_t$. Set $\boldsymbol{y}_0 = \boldsymbol{x}_0$, and it is easy to obtain $\boldsymbol{x}_t = \sum_{s=0}^{t} \boldsymbol{y}_s$. Thus it can equivalently be written as  
$$
\boldsymbol{y}_{t+1} = \boldsymbol{f}_{t+1}\left(\sum_{s=0}^{t} \boldsymbol{y}_s\right)
$$
  
That is, from the perspective of $\boldsymbol{y}$, Residuals obtains $\boldsymbol{y}_{t+1}$ by feeding the equal-weight sum of $\boldsymbol{y}_0,\boldsymbol{y}_1,\cdots,\boldsymbol{y}_t$ into $\boldsymbol{f}_{t+1}$. A natural generalization is to replace that with a weighted sum:  
$$
\boldsymbol{y}_{t+1} = \boldsymbol{f}_{t+1}\left(\sum_{s=0}^{t} a_{t+1,s}\boldsymbol{y}_s\right),\qquad a_{t+1,s}\geq 0,\qquad \sum_{s=0}^{t} a_{t+1,s}=1
$$
  
This was the germ of AttnRes. The equation above also imposes two additional constraints on $a_{t+1,s}$, so let us first discuss why they are needed:

> 1. The constraint $a_{t+1,s}\geq 0$ ensures that the same $\boldsymbol{y}_s$ always contributes in the same direction to different layers, avoiding the inconsistency where one layer wants to increase $\boldsymbol{y}_s$ while another wants to decrease it. Intuitively, this should make optimization easier for the model;
> 
> 2. Our $\boldsymbol{f}$ uses In Norm, so it first applies $\operatorname{RMSNorm}$ to its input. Since $\operatorname{RMSNorm}(\boldsymbol{x})=\operatorname{RMSNorm}(c\boldsymbol{x})$ holds for all $c>0$, weighted averages and weighted sums are completely equivalent, so the constraint $\sum_{s=0}^{t} a_{t+1,s}=1$ does not reduce expressivity.

## Hyper-Connections

Before getting into AttnRes, let us briefly review HC (Hyper-Connections) and show that it too can be understood as inter-layer Attention. This, in turn, suggests that inter-layer Attention is indeed the more fundamental route. HC changes Residuals to  
$$
\begin{equation}\boldsymbol{X}_t = \boldsymbol{H}_t^{res}\boldsymbol{X}_{t-1} + \boldsymbol{H}_t^{post} \boldsymbol{f}_t(\boldsymbol{H}_t^{pre}\boldsymbol{X}_{t-1})\end{equation}
$$
  
where $\boldsymbol{H}_t^{res}\in\mathbb{R}^{k\times k}$, $\boldsymbol{H}_t^{pre}\in\mathbb{R}^{1\times k}$, and $\boldsymbol{H}_t^{post}\in\mathbb{R}^{k\times 1}$; the classical choice is $k=4$. In simple terms, the state variable is widened by a factor of $k$. Before entering $\boldsymbol{f}_t$, an $\boldsymbol{H}_t^{pre}$ matrix maps it back to a single stream; after the output, $\boldsymbol{H}_t^{post}$ expands it back to $k$ streams; and finally it is added to $\boldsymbol{X}_{t-1}$ after mixing by $\boldsymbol{H}_t^{res}$. If we do not restrict the form of $\boldsymbol{H}$, then things like Post Norm and [Highway](https://papers.cool/arxiv/1505.00387) are both special cases of HC.

Similarly, define $\boldsymbol{y}_t=\boldsymbol{f}_t(\boldsymbol{H}_t^{pre}\boldsymbol{X}_{t-1})$, then $\boldsymbol{X}_t = \boldsymbol{H}_t^{res}\boldsymbol{X}_{t-1} + \boldsymbol{H}_t^{post}\boldsymbol{y}_t$. Set $\boldsymbol{X}_0 = \boldsymbol{H}_0^{post}\boldsymbol{y}_0$, and it can be expanded as $\boldsymbol{X}_t = \sum_{s=0}^{t} \boldsymbol{H}_{t\leftarrow s+1}^{res}\boldsymbol{H}_s^{post}\boldsymbol{y}_s$, where $\boldsymbol{H}_{t\leftarrow s}^{res}$ is defined as $\prod_{i=s}^{t}\boldsymbol{H}_i^{res}$. Further define $a_{t+1,s}=\boldsymbol{H}_{t+1}^{pre}\boldsymbol{H}_{t\leftarrow s+1}^{res}\boldsymbol{H}_s^{post}$, and we can write  
$$
\boldsymbol{y}_{t+1} = \boldsymbol{f}_{t+1}\left(\sum_{s=0}^{t} a_{t+1,s}\boldsymbol{y}_s\right)
$$
  
Notice that each $\boldsymbol{H}_{t+1}^{pre}\boldsymbol{H}_{t\leftarrow s+1}^{res}\boldsymbol{H}_s^{post}$ is a $1\times 1$ matrix, i.e. effectively a scalar, so this too is an inter-layer-Attention form of the weighted-sum equation above. Readers familiar with [linear attention](https://kexue.fm/archives/11033) should grasp this result quickly: HC is essentially a “DeltaNet rotated by 90 degrees.” In practice, the three $\boldsymbol{H}$ matrices are produced by simple linear layers with $\tanh$ activation, which means the cumulative $\boldsymbol{H}_{t\leftarrow s}^{res}$ may explode or collapse, and the nonnegativity of $a_{t+1,s}$ cannot be guaranteed.

mHC later improved this. It first changed all three $\boldsymbol{H}$ matrices to Sigmoid activations, guaranteeing $a_{t+1,s}$ is nonnegative. It then alternately normalizes $\boldsymbol{H}_t^{res}$ to make it doubly stochastic; closure of doubly stochastic matrices under multiplication then guarantees the stability of $\boldsymbol{H}_{t\leftarrow s}^{res}$. Experiments also validated the effectiveness of these changes. That said, some newer experiments, such as [“Your DeepSeek mHC May Not Need the ‘m’”](https://zhuanlan.zhihu.com/p/2010852389670908320), suggest that simply setting $\boldsymbol{H}_t^{res}$ to the identity matrix may already be good enough.

## Many Hands Make Light Work

Let us return to AttnRes. Once we realized that AttnRes was feasible, the next question became: what form should $a_{t+1,s}$ take? A natural idea was to follow standard [Scaled Dot-Product Attention](https://kexue.fm/archives/4765), but at the time I wanted a quick first experiment, so I chose a simpler form  
$$
a_{t+1,s}\propto \exp\big(\boldsymbol{w}_{t+1}\cdot \boldsymbol{y}_s\big)
$$
  
where $\boldsymbol{w}_t$ is a trainable vector parameter. In other words, I directly used a data-independent static vector as Q, while both K and V were $\boldsymbol{y}_s$ for a Softmax Attention over depth. This was the first version of AttnRes. Surprisingly, such a simple design already produced a very significant improvement over Residuals!

After I shared the initial AttnRes experiments within the team, [@张宇](https://x.com/yzhang_cs) and [@广宇](https://x.com/nathancgy4) became very interested and joined in. We started validating it on larger-scale models, and the results were consistently delightful. Along the way, we also tried some more complicated designs, but most of them underperformed this simple version. The only modification that delivered a fairly stable gain was to add an extra $\operatorname{RMSNorm}$ to K, which led to the final form of AttnRes  
$$
a_{t+1,s}\propto \exp\big(\boldsymbol{w}_{t+1}\cdot \operatorname{RMSNorm}(\boldsymbol{y}_s)\big)
$$

However, AttnRes is, after all, a dense inter-layer connectivity scheme. Would training and inference remain feasible at K2 and even larger scales? Excitingly, [@V哥](https://zhuanlan.zhihu.com/p/2017528295286133070), after some brilliant analysis, was the first to confirm its feasibility for inference. The real stroke of genius here was precisely the static-Q design that had originally been chosen for convenience! It means that once we compute $\boldsymbol{y}_s$, we can precompute the attention weights $a_{t,s}$ for all $t>s$, which gave Infra enough room to maneuver.

But unfortunately, colleagues on the training side such as [@王哥](https://www.zhihu.com/question/2016993095078684011/answer/2017381145474508331), after careful analysis, concluded that under our current training environment AttnRes was still not feasible enough (to be blunt, we were still limited by resources). We needed a further scheme to reduce communication and memory use, and that led to the Block version below. Correspondingly, we call the earlier version the Full version.

## Blockwise Version

Moving from Full AttnRes to Block AttnRes is analogous to the old process of linearizing quadratic Attention. In principle, many existing Efficient Attention ideas could be tried here as well. Our first attempt, for example, was SWA (Sliding Window Attention), but the actual results were terrible, even worse than Residuals.

After thinking it through, I came to the following view: Residuals itself is already a very strong baseline. It corresponds to the equal-weight sum of all state vectors. Any new design that wants to surpass it must, at the very least, be able to express that case. Full AttnRes clearly satisfies this condition, but once SWA is added it no longer does, because it discards part of the state and therefore cannot cover the special case of “equal-weight summation over all state vectors.”

This made us realize that for AttnRes, “compression” might be more effective than “sparsity,” and the compression need not be very delicate: a simple weighted sum might already be enough. After a round of design and polishing, [@张宇](https://x.com/yzhang_cs) and [@广宇](https://x.com/nathancgy4) proposed the Block AttnRes design used in the paper. It combines blockwise processing with sum-based compression and achieves performance close to the Full version.

The basic idea of Block AttnRes is as follows: first, the Embedding layer is treated as its own block. This is because, by inspecting the Attention matrix of the Full version (which is one of the advantages of the Attention viewpoint: you can visualize attention patterns at any time), we found that the model tends to assign substantial Attention to the Embedding layer, so it is necessary to isolate it. The remaining layers are then divided into blocks of every $m$ layers; within each block they are compressed by summation, and inter-block Attention is computed over these summed representations.

Experiments show that fixing the model to around 8 blocks is enough to recover most of AttnRes’s gains. After evaluation, both the training and inference engineers agreed that the extra overhead of Block AttnRes is very small and entirely worth the improvement in effectiveness (for detailed analyses, see the posts by [@王哥](https://www.zhihu.com/question/2016993095078684011/answer/2017381145474508331) and [@V哥](https://zhuanlan.zhihu.com/p/2017528295286133070); if you want a rough number, it is on the order of less than 5% overhead in exchange for about 25% gain). So everyone pushed hard to get it into the mainline. That was another full and enjoyable stretch of work, and I will not go into it here.

## Matrix View

Worth noting, we can also use the Attention matrix to unify Residuals, HC/mHC, Full AttnRes, and Block AttnRes. This is another rather interesting perspective. A toy example is shown below. Here $\phi(\boldsymbol{q},\boldsymbol{k}) = \exp(\boldsymbol{q}\cdot \operatorname{RMSNorm}(\boldsymbol{k}))$. For the Block AttnRes example, $m=3$, and $\boldsymbol{y}_{s:t} = \sum_{i=s}^{t}\boldsymbol{y}_i$; we also used this latter notation in [“Make Model Alchemy a Bit More Scientific (IV): New Identities, New Learning Rates”](https://kexue.fm/archives/11494).

### Residuals

$$
\boldsymbol{A}=\left(\begin{array}{c} 
1 \\ 
1 & 1 \\ 
1 & 1 & 1 \\ 
1 & 1 & 1 & 1 \\ 
1 & 1 & 1 & 1 & 1 \\ 
1 & 1 & 1 & 1 & 1 & 1 \\ 
1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 
\end{array}\right)
$$

### HC/mHC

$$
\boldsymbol{A}=\left(\begin{array}{c} 
\boldsymbol{H}_1^{pre} \boldsymbol{H}_0^{post} \\ 
\boldsymbol{H}_2^{pre}\boldsymbol{H}_{1\leftarrow 1}^{res}\boldsymbol{H}_0^{post} & \boldsymbol{H}_2^{pre}\boldsymbol{H}_1^{post} \\ 
\boldsymbol{H}_3^{pre}\boldsymbol{H}_{2\leftarrow 1}^{res}\boldsymbol{H}_0^{post} & \boldsymbol{H}_3^{pre}\boldsymbol{H}_{2\leftarrow 2}^{res}\boldsymbol{H}_1^{post} & \boldsymbol{H}_3^{pre}\boldsymbol{H}_2^{post} \\ 
\boldsymbol{H}_4^{pre}\boldsymbol{H}_{3\leftarrow 1}^{res}\boldsymbol{H}_0^{post} & \boldsymbol{H}_4^{pre}\boldsymbol{H}_{3\leftarrow 2}^{res}\boldsymbol{H}_1^{post} & \boldsymbol{H}_4^{pre}\boldsymbol{H}_{3\leftarrow 3}^{res}\boldsymbol{H}_2^{post} & \boldsymbol{H}_4^{pre}\boldsymbol{H}_3^{post} \\ 
\boldsymbol{H}_5^{pre}\boldsymbol{H}_{4\leftarrow 1}^{res}\boldsymbol{H}_0^{post} & \boldsymbol{H}_5^{pre}\boldsymbol{H}_{4\leftarrow 2}^{res}\boldsymbol{H}_1^{post} & \boldsymbol{H}_5^{pre}\boldsymbol{H}_{4\leftarrow 3}^{res}\boldsymbol{H}_2^{post} & \boldsymbol{H}_5^{pre}\boldsymbol{H}_{4\leftarrow 4}^{res}\boldsymbol{H}_3^{post} & \boldsymbol{H}_5^{pre}\boldsymbol{H}_4^{post} \\ 
\boldsymbol{H}_6^{pre}\boldsymbol{H}_{5\leftarrow 1}^{res}\boldsymbol{H}_0^{post} & \boldsymbol{H}_6^{pre}\boldsymbol{H}_{5\leftarrow 2}^{res}\boldsymbol{H}_1^{post} & \boldsymbol{H}_6^{pre}\boldsymbol{H}_{5\leftarrow 3}^{res}\boldsymbol{H}_2^{post} & \boldsymbol{H}_6^{pre}\boldsymbol{H}_{5\leftarrow 4}^{res}\boldsymbol{H}_3^{post} & \boldsymbol{H}_6^{pre}\boldsymbol{H}_{5\leftarrow 4}^{res}\boldsymbol{H}_4^{post} & \boldsymbol{H}_6^{pre}\boldsymbol{H}_5^{post} \\ 
\boldsymbol{H}_7^{pre}\boldsymbol{H}_{6\leftarrow 1}^{res}\boldsymbol{H}_0^{post} & \boldsymbol{H}_7^{pre}\boldsymbol{H}_{6\leftarrow 2}^{res}\boldsymbol{H}_1^{post} & \boldsymbol{H}_7^{pre}\boldsymbol{H}_{6\leftarrow 3}^{res}\boldsymbol{H}_2^{post} & \boldsymbol{H}_7^{pre}\boldsymbol{H}_{6\leftarrow 4}^{res}\boldsymbol{H}_3^{post} & \boldsymbol{H}_7^{pre}\boldsymbol{H}_{6\leftarrow 5}^{res}\boldsymbol{H}_4^{post} & \boldsymbol{H}_7^{pre}\boldsymbol{H}_{6\leftarrow 6}^{res}\boldsymbol{H}_5^{post} & \boldsymbol{H}_7^{pre}\boldsymbol{H}_6^{post} \\ 
\end{array}\right)
$$

### Full AttnRes

$$
\boldsymbol{A}=\left(\begin{array}{c} 
\phi(\boldsymbol{w}_1, \boldsymbol{y}_0) \\ 
\phi(\boldsymbol{w}_2, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_2, \boldsymbol{y}_1) \\ 
\phi(\boldsymbol{w}_3, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_3, \boldsymbol{y}_1) & \phi(\boldsymbol{w}_3, \boldsymbol{y}_2) \\ 
\phi(\boldsymbol{w}_4, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_4, \boldsymbol{y}_1) & \phi(\boldsymbol{w}_4, \boldsymbol{y}_2) & \phi(\boldsymbol{w}_4, \boldsymbol{y}_3) \\ 
\phi(\boldsymbol{w}_5, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_5, \boldsymbol{y}_1) & \phi(\boldsymbol{w}_5, \boldsymbol{y}_2) & \phi(\boldsymbol{w}_5, \boldsymbol{y}_3) & \phi(\boldsymbol{w}_5, \boldsymbol{y}_4) \\ 
\phi(\boldsymbol{w}_6, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_1) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_2) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_3) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_4) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_5) \\ 
\phi(\boldsymbol{w}_7, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_1) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_2) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_3) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_4) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_5) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_6) \\ 
\end{array}\right)
$$

### Block AttnRes

$$
\boldsymbol{A}=\left(\begin{array}{c:ccc:ccc} 
\phi(\boldsymbol{w}_1, \boldsymbol{y}_0) \\ 
\hdashline 
\phi(\boldsymbol{w}_2, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_2, \boldsymbol{y}_1) \\ 
\phi(\boldsymbol{w}_3, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_3, \boldsymbol{y}_{1:2}) & \phi(\boldsymbol{w}_3, \boldsymbol{y}_{1:2}) \\ 
\phi(\boldsymbol{w}_4, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_4, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_4, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_4, \boldsymbol{y}_{1:3}) \\ 
\hdashline 
\phi(\boldsymbol{w}_5, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_5, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_5, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_5, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_5, \boldsymbol{y}_4)\\ 
\phi(\boldsymbol{w}_6, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_{4:5}) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_{4:5}) \\ 
\phi(\boldsymbol{w}_7, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_{4:6}) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_{4:6}) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_{4:6}) \\ 
\end{array}\right)
$$

## Related Work

From the moment we decided to work on AttnRes, I and many teammates were immersed in polishing, validating, and accelerating it. Some readers may know that my usual research style is to first push derivation and problem-solving as far as I possibly can, and only when I hit difficulty or fully solve the problem do I start looking for related literature. By chance, I happened to be surrounded this time by a group of like-minded collaborators, and by chance the overall AttnRes exploration went rather smoothly. So it was only after essentially all tests had passed and we began preparing the technical report that we started a literature review.

And precisely because of that, once we looked, we were startled by how much already existed: there had already been a great deal of work related to Dense Connection and Depth Attention. Besides the classic [DenseNet](https://papers.cool/arxiv/1608.06993), the works we found included [DenseFormer](https://papers.cool/arxiv/2402.02622), [ANCRe](https://papers.cool/arxiv/2602.09009), [MUDDFormer](https://papers.cool/arxiv/2502.12170), [MRLA](https://papers.cool/arxiv/2302.03985), [Dreamer](https://papers.cool/arxiv/2601.21582), and more. Even pre-BERT [ELMo](https://papers.cool/arxiv/1802.05365) had partially used a similar design. We included all of these in the references.

After the technical report was released, we gradually received comments from readers pointing out additional related work that we had not yet included, such as [SKNets](https://papers.cool/arxiv/1903.06586), [LIMe](https://papers.cool/arxiv/2502.09245), [DCA](https://papers.cool/arxiv/2502.06785), and others. For this we apologize and are grateful, and we promise to add as many of them as possible in later revisions. But whether you are a reader or an author, please stay rational about this: literature review is not easy, and some omissions are unavoidable. We hold all related work in the highest respect.

At the same time, we also hope people will pay more attention to the work involved in AttnRes beyond the concept of “Depth Attention” itself. We fully agree that in 2026, “Depth Attention,” or “Layer Attention,” is not a novel idea at all. But making it work on sufficiently large models, turning it into a strong enough replacement for Residuals, and at the same time satisfying the efficiency requirements of both training and inference is far from easy. To the best of our knowledge, AttnRes is the first work to achieve that.

## Summary

This article introduced our latest architectural result, Attention Residuals (AttnRes), which replaces naive Residuals with inter-layer Attention and, through careful design, meets the efficiency requirements of both training and inference, ultimately making it possible to scale the method to sufficiently large models.

***If you repost this article, please include this article’s URL:** [https://kexue.fm/archives/11664](https://kexue.fm/archives/11664 "A Memoir of Attention Residuals")*

***For more detailed reposting instructions, please see:*** [“Scientific Spaces FAQ”](https://kexue.fm/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8 "Scientific Spaces FAQ")

**If you still have any questions or suggestions, feel free to continue the discussion in the comments section below.**

**If you think this article was pretty good, you are welcome to [share](https://kexue.fm/archives/11664#share) / [tip](https://kexue.fm/archives/11664#pay) it. The point of tipping is not to make money from it, but to let us know how many readers truly care about Scientific Spaces. Of course, ignoring it will not affect your reading in any way. Again, thank you very much, and you are most welcome!**

**If you need to cite this article, please use:**

Jianlin Su. (Mar. 19, 2026). *A Memoir of Attention Residuals* \[Blog post\]. Retrieved from [https://kexue.fm/archives/11664](https://kexue.fm/archives/11664)

@online{kexuefm-11664,  
title={A Memoir of Attention Residuals},  
author={Jianlin Su},  
year={2026},  
month={Mar},  
url={\\url{https://kexue.fm/archives/11664}},  
}

Category: [Information Age](https://kexue.fm/category/Big-Data) Tags: ,,,,

< [A Muon implementation idea based on streaming power iteration](https://kexue.fm/archives/11654 "A Muon implementation idea based on streaming power iteration") | \>

### You may also be interested in the following

- [A Tour of MoE: 7. The minimalist solution to dynamic activation](https://kexue.fm/archives/11626 "A Tour of MoE: 7. The minimalist solution to dynamic activation")
- [A Tour of MoE: 6. Optimal allocation for better balance](https://kexue.fm/archives/11619 "A Tour of MoE: 6. Optimal allocation for better balance")
- [The entries of DeltaNet’s core inverse matrix are always in \[-1, 1\]](https://kexue.fm/archives/11563 "The entries of DeltaNet's core inverse matrix are always in [-1, 1]")
- [Make model alchemy a bit more scientific (VI): an exquisite top-down construction](https://kexue.fm/archives/11540 "Make model alchemy a bit more scientific (VI): an exquisite top-down construction")
- [Why does DeltaNet need L2 Normalize?](https://kexue.fm/archives/11486 "Why does DeltaNet need L2 Normalize?")
- [A guide to the Muon optimizer: quick start and key details](https://kexue.fm/archives/11416 "A guide to the Muon optimizer: quick start and key details")
- [Low-precision Attention may have biased rounding errors](https://kexue.fm/archives/11371 "Low-precision Attention may have biased rounding errors")
- [Beyond MuP: 1. Three characteristics of a good model](https://kexue.fm/archives/11340 "Beyond MuP: 1. Three characteristics of a good model")
- [Why add Short Conv to linear attention?](https://kexue.fm/archives/11320 "Why add Short Conv to linear attention?")
- [Rethinking learning rate and batch size (IV): EMA](https://kexue.fm/archives/11301 "Rethinking learning rate and batch size (IV): EMA")
