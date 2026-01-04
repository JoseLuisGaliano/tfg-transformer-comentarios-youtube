import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import argparse

def group_train_val_test_split_balanced(
    df, group_col, label_col,
    train_size=0.7, val_size=0.15, test_size=0.15,
    n_repeats=50, seed=42
):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Las proporciones deben sumar 1."
    classes = set(df[label_col].unique())
    rng = np.random.RandomState(seed)

    def all_classes_present(subdf):
        return set(subdf[label_col].unique()) == classes

    maxseed = seed + n_repeats
    best_counts = float('inf')
    best_split = None

    for s in range(seed, maxseed):
        # 1) Split test
        gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size+val_size, test_size=test_size, random_state=s)
        idx = np.arange(len(df))
        groups = df[group_col]
        trainval_idx, test_idx = next(gss1.split(idx, groups=groups))
        df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        # 2) Split val
        val_rel = val_size / (train_size + val_size)
        gss2 = GroupShuffleSplit(n_splits=1, train_size=1-val_rel, test_size=val_rel, random_state=s)
        idx_tv = np.arange(len(df_trainval))
        groups_tv = df_trainval[group_col]
        train_idx, val_idx = next(gss2.split(idx_tv, groups=groups_tv))
        df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
        df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

        # 3) Condiciones de balance
        train_ok = all_classes_present(df_train)
        val_ok = all_classes_present(df_val)
        test_ok = all_classes_present(df_test)

        split_unbalanced_count = int(not train_ok) + int(not val_ok) + int(not test_ok)
        if split_unbalanced_count < best_counts:
            best_counts = split_unbalanced_count
            best_split = (df_train.copy(), df_val.copy(), df_test.copy())
            if best_counts == 0:
                break  # ya está perfecto

    if best_split is None:
        raise RuntimeError("No se pudo crear ningún split (esto no debería ocurrir).")

    if best_counts > 0:
        print("¡Advertencia! No se pudo lograr un split perfecto con todas las clases presentes en cada split tras múltiples intentos.")
    else:
        print("¡Split balanceado encontrado con todas las clases presentes!")

    return best_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_in", type=str, help="Ruta al CSV completo de entrada")
    parser.add_argument("--group_col", type=str, default="video_id")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--train_size", type=float, default=0.7)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--n_repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_out", type=str, default="train.csv")
    parser.add_argument("--val_out", type=str, default="val.csv")
    parser.add_argument("--test_out", type=str, default="test.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_in)
    df = df[[args.group_col, args.label_col] + [c for c in df.columns if c not in [args.group_col, args.label_col]]]
    df = df.dropna(subset=[args.group_col, args.label_col]).reset_index(drop=True)

    train, val, test = group_train_val_test_split_balanced(
        df,
        group_col=args.group_col,
        label_col=args.label_col,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        n_repeats=args.n_repeats,
        seed=args.seed
    )

    train.to_csv(args.train_out, index=False)
    val.to_csv(args.val_out, index=False)
    test.to_csv(args.test_out, index=False)
    print("¡Splits guardados como CSV!")

