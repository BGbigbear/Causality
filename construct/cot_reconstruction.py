import json


def cot_construct():
    with (
        open("../data/reconstruction/train2_e.json", "r", encoding='utf-8') as f_raw,
        open("../data/reconstruction/train2_cot.json", "w", encoding='utf-8') as f_cot,
    ):
        data = json.load(f_raw)
        cot_data = []
        for doc in data:
            new_list = []
            for c_list in doc['causality_list']:
                cause, effect = c_list['cause_event'], c_list['effect_event']
                cause_desc = (f"{cause['actor']}{"于"+cause['time'] if cause['time'] else ""}"
                              f"{"在"+cause['location'] if cause['location'] else ""}"
                              f"{cause['action']}{cause['object']}")
                effect_desc = (f"{effect['actor']}{"于"+effect['time'] if effect['time'] else ""}"
                               f"{"在"+effect['location'] if effect['location'] else ""}"
                               f"{effect['action']}{effect['object']}")
                full_desc = f"{cause_desc}{c_list['causality_type']}导致{effect_desc}"
                cause_event = {"event_description": cause_desc}
                cause_event.update(cause)
                effect_event = {"event_description": effect_desc}
                effect_event.update(effect)
                new_list.append({
                    "causality_description": full_desc,
                    "causality_type": c_list['causality_type'],
                    "cause_event": cause_event,
                    "effect_event": effect_event
                })
            cot_data.append({
                "document_id": doc['document_id'],
                "text": doc['text'],
                "causality_list": new_list
            })

        json.dump(cot_data, f_cot, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    cot_construct()
