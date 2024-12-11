from dataclasses import dataclass

@dataclass
class TaskOrder:
    first = {
    "AboveBelow": [
        'AboveBelow7.json',
        'AboveBelow3.json',
        'AboveBelow6.json',
        'AboveBelow8.json',
        'AboveBelow2.json'
    ],
    "Center": [
        'Center3.json',
        'Center6.json',
        'Center5.json',
        'Center4.json',
        'Center7.json'
    ],
    "ExtendToBoundary": [
        'ExtendToBoundary3.json',
        'ExtendToBoundary4.json',
        'ExtendToBoundary2.json',
        'ExtendToBoundary9.json',
        'ExtendToBoundary6.json'
    ],
    "InsideOutside": [
        'InsideOutside5.json',
        'InsideOutside1.json',
        'InsideOutside3.json',
        'InsideOutside7.json',
        'InsideOutside2.json'
    ],
    "SameDifferent": [
        'SameDifferent3.json',
        'SameDifferent1.json',
        'SameDifferent7.json',
        'SameDifferent6.json',
        'SameDifferent9.json'
    ]
    }
    second = {
        "AboveBelow": [
            'AboveBelow2.json',
            'AboveBelow3.json',
            'AboveBelow6.json',
            'AboveBelow8.json',
            'AboveBelow7.json'
        ],
        "Center": [
            'Center7.json',
            'Center6.json',
            'Center5.json',
            'Center4.json',
            'Center3.json'
        ],
        "ExtendToBoundary": [
            'ExtendToBoundary6.json',
            'ExtendToBoundary4.json',
            'ExtendToBoundary2.json',
            'ExtendToBoundary9.json',
            'ExtendToBoundary3.json'
        ],
        "InsideOutside": [
            'InsideOutside2.json',
            'InsideOutside1.json',
            'InsideOutside3.json',
            'InsideOutside7.json',
            'InsideOutside5.json'
        ],
        "SameDifferent": [
            'SameDifferent9.json',
            'SameDifferent1.json',
            'SameDifferent7.json',
            'SameDifferent6.json',
            'SameDifferent3.json'
        ]
    };


