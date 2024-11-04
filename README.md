# Causality

Extraction the causality of a text

An example:
```text
text: 据印度媒体19日报道，印度北部锡金邦18日18时10分发生里氏6.8级地震，震源深度约20公里。据印度媒体报道，锡金邦首府甘托克地区电路，交通等设施遭到严重破坏。由于断电和设施等问题，甘托克陷入一片黑暗，手机信号和网络中断。地震发生后，又接连发生两次强烈余震。
```
```json
{
    "causality_list": [
        {
            "causality_type": "直接",				
            "cause": {
                "actor": "印度北部锡金邦",
                "class": "安全事件",
                "action": "发生",
                "time": "18日18时10分",
                "location": "印度北部锡金邦",
                "object": "里氏6.8级地震"
            },
            "effect": {
                "actor": "电路，交通等设施",
                "class": "社会事件",
                "action": "遭到",
                "time": "",
                "location": "甘托克",
                "object": "严重破坏"
            }
        },
        {
            "causality_type": "直接",
            "cause": {
                "actor": "电路，交通等设施",
                "class": "社会事件",
                "action": "遭到",
                "time": "",
                "location": "甘托克",
                "object": "严重破坏"
            },
            "effect": {
                "actor": "甘托克",	
                "class": "社会事件",
                "action": "陷入",
                "time": "",
                "location": "甘托克",
                "object": "一片黑暗，手机信号和网络中断"
            }
        }
    ]
}
```
